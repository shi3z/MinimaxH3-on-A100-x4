"""h3_sla: per-job thu-ml/SLA block-sparse attention for MiniMax-H3 (node-driven).

The matched FAST sparse patch for the lightx2v SLA turbo LoRA
(minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16). That LoRA has NO proj_l
-> it was trained against a PURE block-sparse attention (SLA's sparse branch), so
this runs thu-ml/SLA's get_block_map (mean-pooled q.k top-k key blocks) + Triton
_attention kernel. ~2.46x vs FA-2 on attention at S~40868 (A100, bf16); ~1.3-1.7x
end-to-end at 65-85% sparsity on the 4-step fl2v turbo path.

Node `H3SLAPatch(model, topk, blk, enabled) -> MODEL`: sets a per-model
`transformer_options['optimized_attention_override']` (ComfyUI's native hook, the
one MiniMax-H3-Longvideos' sparse_attention_active checks) with topk baked into the
closure. Insert it on the MODEL chain BEFORE the sampler/guider. Per-prompt: two
graphs in the same process can use different topk (or none). topk = 1 - sparsity
(0.35 = 65% sparse, the recommended sweet spot; 0.15 = 85%).

Scope: only the 50 DiT-block self-attentions. A global Attention.forward patch sets
a thread-local flag from `rope_freqs` (DiT blocks pass it; TokenRefiner passes None
-> stays dense). The override is a no-op unless (flag on AND mask is None AND this
model was patched by the node) -> models without the node are byte-identical to stock.

Per-layer presets (2026-09, layer-selective SLA): `preset` = off | quality | balanced | speed | custom.
  off      : previous behaviour, uniform `topk` on all 50 blocks (DEFAULT - nothing changes unless requested)
  quality  : SLA25 on 31-42,46-49, SLA15 elsewhere  (research name E8)
  balanced : SLA25 on 31-42,48-49, SLA15 elsewhere  (research name T2)
  speed    : SLA25 on 34-42,       SLA15 elsewhere  (research name E3)
  custom   : `layer_keep` spec, e.g. "25:31-42,46-49;20:43"  (unlisted layers = `topk`)
PRESETS below is the single source of truth (sp_runtime and the fleet worker only reference it).
Research record: /mnt/ssd/project/newh3/experiments/h3_token_residual/PRESETS.md
Presets were validated on the ref2va turbo 4-step path only; no predictors / memory modules are involved.
"""
import sys
import threading

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_SLA_REPO = "/mnt/ssdraid/project/h3-opt/SLA_repo"
_state = threading.local()
_KFN = {"get_block_map": None, "attention": None}

# ---------------- per-layer keep presets: SINGLE SOURCE OF TRUTH ----------------
# layers: {keep_ratio: "ranges"}; every layer not listed uses default_keep.
PRESETS = {
    "quality":  {"default_keep": 0.15, "layers": {0.25: "31-42,46-49"}, "research_name": "E8"},
    "balanced": {"default_keep": 0.15, "layers": {0.25: "31-42,48-49"}, "research_name": "T2"},
    "speed":    {"default_keep": 0.15, "layers": {0.25: "34-42"},       "research_name": "E3"},
}
PRESET_CHOICES = ["off", "quality", "balanced", "speed", "custom"]
# measured on A100 SP-2 (GPU pair), ref2va turbo 4-step, 1280x720x158 (S~42k): full-generation seconds
# contributed per layer at each keep (least-squares fit, rmse 0.23 s) and attention-core ms per layer per step.
LATENCY_S_PER_LAYER = {0.15: 0.509, 0.20: 0.583, 0.25: 0.662}
ATTN_MS_PER_LAYER_STEP = {0.15: 63.0, 0.25: 101.0, 1.0: 150.0}
STATUS_DIR = "/mnt/ssdraid/project/h3-opt/sp_status/h3_sla"


def parse_ranges(spec):
    """'31-42,46,48-49' -> [31..42, 46, 48, 49]"""
    out = []
    for g in filter(None, (x.strip() for x in str(spec).split(","))):
        a, _, b = g.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def parse_layer_keep(spec):
    """'25:31-42,46-49;20:43' (percent or ratio) -> {layer: keep_ratio}"""
    m = {}
    for part in filter(None, (x.strip() for x in str(spec).split(";"))):
        k, _, rngs = part.partition(":")
        kv = float(k); kv = kv / 100.0 if kv > 1.0 else kv
        for l in parse_ranges(rngs):
            m[l] = kv
    return m


def resolve_schedule(preset="off", topk=0.35, layer_keep="", n_layers=50):
    """-> (name, [keep per layer]).  preset 'off' reproduces the old uniform-topk behaviour exactly."""
    preset = (preset or "off").lower()
    if preset == "off":
        return "off", [float(topk)] * n_layers
    if preset == "custom":
        keeps = [float(topk)] * n_layers
        for l, kv in parse_layer_keep(layer_keep).items():
            if 0 <= l < n_layers:
                keeps[l] = kv
        return "custom", keeps
    if preset not in PRESETS:
        raise ValueError(f"unknown SLA preset {preset!r}; choose one of {PRESET_CHOICES}")
    P = PRESETS[preset]
    keeps = [float(P["default_keep"])] * n_layers
    for kv, rngs in P["layers"].items():
        for l in parse_ranges(rngs):
            if 0 <= l < n_layers:
                keeps[l] = float(kv)
    return preset, keeps


def _ranges_str(ls):
    ls = sorted(ls); out = []
    i = 0
    while i < len(ls):
        j = i
        while j + 1 < len(ls) and ls[j + 1] == ls[j] + 1:
            j += 1
        out.append(f"{ls[i]}-{ls[j]}" if j > i else f"{ls[i]}"); i = j + 1
    return ",".join(out)


def schedule_summary(name, keeps, blk=64):
    counts = {}
    for kv in keeps:
        counts[f"{round(kv * 100)}"] = counts.get(f"{round(kv * 100)}", 0) + 1
    default = max(set(keeps), key=keeps.count)
    by = {f"SLA{round(kv * 100)}": _ranges_str([l for l, x in enumerate(keeps) if x == kv]) for kv in sorted(set(keeps)) if kv != default}
    pred = None
    if all(kv in LATENCY_S_PER_LAYER for kv in keeps):
        pred = round(sum(LATENCY_S_PER_LAYER[kv] for kv in keeps), 2)
    attn = None
    if all(kv in ATTN_MS_PER_LAYER_STEP for kv in keeps):
        attn = round(sum(ATTN_MS_PER_LAYER_STEP[kv] for kv in keeps), 1)
    return {"preset": name, "research_name": PRESETS.get(name, {}).get("research_name"), "default_keep": default,
            "layers_by_keep": by, "counts": counts, "keeps": [round(x, 4) for x in keeps], "blk": blk,
            "predicted_gen_s_sp2_720p158": pred, "predicted_attn_ms_per_step_sp2": attn}


def format_summary(sm):
    lines = [f"SLA preset: {sm['preset']}" + (f" (research {sm['research_name']})" if sm.get("research_name") else ""),
             f"default keep: {round(sm['default_keep'] * 100)}%"]
    for k, r in sm["layers_by_keep"].items():
        lines.append(f"{k} layers: {r}")
    lines.append("layer counts: " + ", ".join(f"SLA{k}={v}" for k, v in sorted(sm["counts"].items())))
    if sm.get("predicted_attn_ms_per_step_sp2") is not None:
        lines.append(f"predicted attention cost: {sm['predicted_attn_ms_per_step_sp2']:.0f} ms/step (SP-2 A100, S~42k)"
                     + (f", gen ~{sm['predicted_gen_s_sp2_720p158']:.1f} s (1280x720x158, 4-step)" if sm.get("predicted_gen_s_sp2_720p158") else ""))
    return "\n".join("[h3_sla] " + x for x in lines)


class _Schedule:
    """Per-model schedule object carried on the override closure (`override.h3sla`).
    run(layer, q, k, v, transformer_options) -> [1,H,S,D] sparse output, or None (=> caller uses dense FA)."""

    def __init__(self, name, keeps, blk):
        import os
        self.name, self.keeps, self.blk = name, keeps, blk
        self.summary = schedule_summary(name, keeps, blk)
        self.timing = os.environ.get("H3_SLA_TIMING") == "1"
        self._ev = []; self.attn_ms = [0.0] * len(keeps); self.attn_n = [0] * len(keeps); self._gen = 0

    def keep_for(self, layer):
        if layer is None or not (0 <= layer < len(self.keeps)):
            return self.summary["default_keep"]
        return self.keeps[layer]

    def _generation_start(self, transformer_options):
        """layer 0 at the first sigma of a sampling run -> log + dashboard status."""
        try:
            sig = transformer_options.get("sigmas"); ss = transformer_options.get("sample_sigmas")
            first = sig is None or ss is None or float(sig.flatten()[0]) >= float(ss.flatten()[0]) - 1e-6
        except Exception:
            first = True
        if not first:
            return
        self._flush_timing()
        self._gen += 1
        import os
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(format_summary(self.summary), flush=True)
        self._write_status()

    def _flush_timing(self):
        if not self._ev:
            return
        import torch
        try:
            self._ev[-1][2].synchronize()
            for l, a, b in self._ev:
                self.attn_ms[l] += a.elapsed_time(b); self.attn_n[l] += 1
        except Exception:
            pass
        self._ev = []
        self._write_status()

    def _write_status(self):
        import json, os, socket, time
        try:
            os.makedirs(STATUS_DIR, exist_ok=True)
            st = dict(self.summary)
            st.update({"host": socket.gethostname(), "pid": os.getpid(), "rank": int(os.environ.get("RANK", "0")),
                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "generation": self._gen,
                       "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "attn_ms_per_layer_call": [round(m / n, 3) if n else None for m, n in zip(self.attn_ms, self.attn_n)] if self.timing else None})
            f = os.path.join(STATUS_DIR, f"{st['host']}_{st['pid']}.json")
            json.dump(st, open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f)
        except Exception as e:
            print(f"[h3_sla] status write failed: {e}", flush=True)

    def run(self, layer, q, k, v, transformer_options=None):
        import torch
        if layer == 0:
            self._generation_start(transformer_options or {})
        keep = self.keep_for(layer)
        if keep >= 1.0:
            return None
        if self.timing and layer is not None:
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True); a.record()
            o = _sla_heads(q, k, v, keep, self.blk); b.record(); self._ev.append((layer, a, b))
            if layer == len(self.keeps) - 1:          # end of a denoise step: resolve events, update dashboard status
                self._flush_timing()
            return o
        return _sla_heads(q, k, v, keep, self.blk)


def _load_kernel():
    if _KFN["attention"] is None:
        if _SLA_REPO not in sys.path:
            sys.path.insert(0, _SLA_REPO)
        from sparse_linear_attention.utils import get_block_map
        from sparse_linear_attention.kernel import _attention
        _KFN["get_block_map"] = get_block_map
        _KFN["attention"] = _attention
    return _KFN


def _sla_heads(q, k, v, topk, blk):
    """q,k,v: [1,H,S,D] -> [1,H,S,D] in q.dtype (SLA sparse branch only; K/V may be the unpadded length)."""
    import torch
    kfn = _load_kernel()
    out_dtype = q.dtype
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    sparse_map, lut, real_topk = kfn["get_block_map"](q, k, topk_ratio=topk, BLKQ=blk, BLKK=blk)
    o_s = kfn["attention"].apply(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                                 sparse_map, lut, real_topk, blk, blk)  # [1,H,S,D]
    return o_s.to(out_dtype)


def _sla_sparse_attention(q, k, v, topk, blk):
    """q,k,v: [1,H,S,D] (skip_reshape). Returns [1,S,H*D] (SLA sparse branch only)."""
    _, H, S, D = q.shape
    return _sla_heads(q, k, v, topk, blk).transpose(1, 2).reshape(1, S, H * D)


def _install_scope_patch():
    """Global, harmless: mark DiT-block attention via rope_freqs (idempotent)."""
    import comfy.ldm.minimax.model as mm
    if getattr(mm.Attention.forward, "_h3sla_wrapped", False):
        return
    orig = mm.Attention.forward

    def patched(self, x, rope_freqs=None, transformer_options={}):
        prev = getattr(_state, "on", False); prev_l = getattr(_state, "layer", None)
        _state.on = rope_freqs is not None
        _state.layer = getattr(self, "_h3sla_layer", None)
        try:
            return orig(self, x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        finally:
            _state.on = prev; _state.layer = prev_l

    patched._h3sla_wrapped = True
    mm.Attention.forward = patched
    print("[h3_sla] DiT-scope patch installed (node-driven; inert until H3SLAPatch used)", flush=True)


def _make_override(topk, blk, schedule=None):
    """schedule=None: uniform topk (legacy). Otherwise a _Schedule (per-layer keep). The schedule is also
    exposed as `override.h3sla` so the Ulysses SP attention (h3-opt sp_runtime) applies the same schedule."""
    def override(func, *args, **kwargs):
        # args = (q, k, v, heads, ...); kwargs may hold mask/skip_reshape/transformer_options
        try:
            if getattr(_state, "on", False) and kwargs.get("mask") is None and len(args) >= 3:
                q, k, v = args[0], args[1], args[2]
                if hasattr(q, "shape") and q.dim() == 4 and q.shape[-1] == 128:
                    if schedule is None:
                        return _sla_sparse_attention(q, k, v, topk, blk)
                    o = schedule.run(getattr(_state, "layer", None), q, k, v, kwargs.get("transformer_options"))
                    if o is not None:
                        _, H, S, D = q.shape
                        return o.transpose(1, 2).reshape(1, S, H * D)
        except Exception as e:  # never break a render on a sparse path
            print(f"[h3_sla] override fell back to dense: {e}", flush=True)
        return func(*args, **kwargs)
    override.h3sla = schedule      # None for the legacy uniform mode -> sp_runtime keeps its old (dense FA2) path
    return override


def get_schedule(transformer_options):
    """For other attention implementations (sp_runtime): the active _Schedule or None."""
    try:
        ov = (transformer_options or {}).get("optimized_attention_override")
        return getattr(ov, "h3sla", None)
    except Exception:
        return None


def _tag_layers(model_patcher):
    """Record each DiT block's index on its attention module (read by the scope patch / sp_runtime)."""
    try:
        blocks = model_patcher.model.diffusion_model.blocks
        for i, b in enumerate(blocks):
            b.attn._h3sla_layer = i
        return len(blocks)
    except Exception as e:
        print(f"[h3_sla] could not tag DiT layers ({e}); per-layer presets need MiniMax-H3 blocks", flush=True)
        return None


class H3SLAPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "topk": ("FLOAT", {"default": 0.35, "min": 0.02, "max": 1.0, "step": 0.01,
                               "tooltip": "keys kept = 1 - sparsity. 0.35=65% sparse (sweet spot), 0.15=85%"}),
            "blk": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
            "enabled": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "preset": (PRESET_CHOICES, {"default": "off",
                       "tooltip": "off = uniform topk (unchanged legacy behaviour). quality/balanced/speed = validated "
                                  "per-layer schedules (SLA25 on a few layers, SLA15 elsewhere; topk ignored). custom = layer_keep."}),
            "layer_keep": ("STRING", {"default": "", "tooltip": "custom only: '25:31-42,46-49;20:43' (percent:layer ranges); other layers use topk"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MiniMaxH3"

    def patch(self, model, topk, blk, enabled, preset="off", layer_keep=""):
        if not enabled:
            return (model,)
        _install_scope_patch()
        _load_kernel()  # fail loudly here, not mid-render
        m = model.clone()
        m.model_options = dict(m.model_options)
        m.model_options["transformer_options"] = dict(m.model_options.get("transformer_options", {}))
        preset = (preset or "off").lower()
        if preset == "off":
            m.model_options["transformer_options"]["optimized_attention_override"] = _make_override(float(topk), int(blk))
            print(f"[h3_sla] H3SLAPatch: topk={topk:.3f} (sparsity={100*(1-topk):.0f}%) blk={blk}", flush=True)
            return (m,)
        n = _tag_layers(m) or 50
        name, keeps = resolve_schedule(preset, float(topk), layer_keep, n)
        sched = _Schedule(name, keeps, int(blk))
        m.model_options["transformer_options"]["optimized_attention_override"] = _make_override(float(topk), int(blk), sched)
        print(f"[h3_sla] H3SLAPatch: per-layer schedule blk={blk}\n" + format_summary(sched.summary), flush=True)
        return (m,)


NODE_CLASS_MAPPINGS["H3SLAPatch"] = H3SLAPatch
NODE_DISPLAY_NAME_MAPPINGS["H3SLAPatch"] = "H3 SLA Sparse Attention"

# install the scope patch at import so it's ready (harmless; inert without a patched model)
try:
    _install_scope_patch()
except Exception as e:
    print(f"[h3_sla] scope patch deferred (comfy not ready): {e}", flush=True)
