"""H3 denoiser profiler — env-gated, non-invasive.

Activates ONLY when H3_PROFILE=1, so it is inert for normal fleet runs.
Wraps the MiniMax H3 DiT forward + sub-modules with record_function ranges and
CUDA-event timers to produce a per-component breakdown of ONE denoise step.

Outputs to /mnt/ssdraid/project/h3-opt/logs/:
  - profile_step.json   (per-component CUDA ms, summed over all 50 blocks, for the target step)
  - profile_steps.jsonl (total forward ms for every step via CUDA events)
  - profile_trace.txt   (torch.profiler key_averages table for the target step)
Env:
  H3_PROFILE=1        enable
  H3_PROFILE_STEP=N   which denoise-forward index to deep-profile (default 2, after warmups)
"""
import os, json, time
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if os.environ.get("H3_PROFILE") == "1":
    import torch
    from torch.profiler import record_function
    LOGDIR = "/mnt/ssdraid/project/h3-opt/logs"
    os.makedirs(LOGDIR, exist_ok=True)
    TARGET = int(os.environ.get("H3_PROFILE_STEP", "2"))

    import comfy.ldm.minimax.model as M

    _state = {"call": 0, "armed_dump": False}

    # ---- coarse component ranges: wrap whole sub-module forwards in record_function ----
    _orig_attn = M.Attention.forward
    _orig_mlp = M.MLP.forward
    _orig_adaln = M.AdalnProj.forward
    _orig_block = M.DiTBlock.forward
    _orig_optattn = M.optimized_attention

    def attn_fwd(self, *a, **k):
        with record_function("H3.attn"):
            return _orig_attn(self, *a, **k)
    def mlp_fwd(self, *a, **k):
        with record_function("H3.ffn"):
            return _orig_mlp(self, *a, **k)
    def adaln_fwd(self, *a, **k):
        with record_function("H3.adaln"):
            return _orig_adaln(self, *a, **k)
    _dump = {"want": os.environ.get("H3_DUMP") == "1", "blk": 0, "done": False, "nblocks": 50}
    def block_fwd(self, *a, **k):
        # capture block-loop I/O on the target step for offline SP verification
        if _dump["want"] and _state["armed_dump"] and not _dump["done"]:
            bi = _dump["blk"]; _dump["blk"] += 1
            x = a[0] if a else k.get("x")
            if bi == 0:
                t_emb = a[1] if len(a) > 1 else k.get("t_emb")
                mod_segments = a[2] if len(a) > 2 else k.get("mod_segments")
                rope_freqs = a[3] if len(a) > 3 else k.get("rope_freqs")
                torch.save({"h_in": x.detach().to("cpu"), "t_emb": t_emb.detach().to("cpu"),
                            "mod_segments": list(mod_segments), "rope_freqs": rope_freqs.detach().to("cpu"),
                            "num_blocks": _dump["nblocks"]}, f"{LOGDIR}/io_blockloop_in.pt")
                print(f"[h3_profiler] captured block-loop IN h={tuple(x.shape)} segs={len(mod_segments)} rope={tuple(rope_freqs.shape)}", flush=True)
            out = _orig_block(self, *a, **k)
            if bi == _dump["nblocks"] - 1:
                torch.save({"h_out": out.detach().to("cpu")}, f"{LOGDIR}/io_blockloop_out.pt")
                _dump["done"] = True; _state["armed_dump"] = False
                print(f"[h3_profiler] captured block-loop OUT h={tuple(out.shape)}", flush=True)
            return out
        with record_function("H3.block"):
            return _orig_block(self, *a, **k)
    def optattn_fwd(*a, **k):
        with record_function("H3.softmax"):
            return _orig_optattn(*a, **k)

    M.Attention.forward = attn_fwd
    M.MLP.forward = mlp_fwd
    M.AdalnProj.forward = adaln_fwd
    M.DiTBlock.forward = block_fwd
    M.optimized_attention = optattn_fwd

    # ---- fine CUDA-event component accumulation for the target step ----
    # buckets accumulate (start_evt, end_evt) pairs across all 50 blocks, summed after one sync.
    _prof = {"armed": False, "ev": {}}
    def _bucket(name):
        return _prof["ev"].setdefault(name, [])
    def _mark(name, fn, *a, **k):
        if not _prof["armed"]:
            return fn(*a, **k)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); r = fn(*a, **k); e.record()
        _bucket(name).append((s, e))
        return r

    # finer attention reimplementation mirroring model.py Attention.forward, with markers
    def attn_fwd_fine(self, x, rope_freqs=None, transformer_options={}):
        if not _prof["armed"]:
            return attn_fwd(self, x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        s = x.shape[0]
        qkv = _mark("attn.qkv", self.qkv_proj, x)
        q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            q = q.view(1, s, self.heads, self.head_dim); k = k.view(1, s, self.heads, self.head_dim)
            qw = M.comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = M.comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            _mark("attn.qknorm_rope", M.comfy.quant_ops.ck.rms_rope_split_half_,
                  q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            q = q[0]; k = k[0]
        else:
            q = self.q_norm(q.view(s, self.heads, self.head_dim)); k = self.k_norm(k.view(s, self.heads, self.head_dim))
        q = q.transpose(0, 1).unsqueeze(0); k = k.transpose(0, 1).unsqueeze(0); v = v.transpose(0, 1).unsqueeze(0)
        out = _mark("attn.softmax", _orig_optattn, q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
        return _mark("attn.out", self.out_proj, out.squeeze(0))

    def mlp_fwd_fine(self, x):
        if not _prof["armed"]:
            return mlp_fwd(self, x)
        h1 = _mark("ffn.fc1", self.fc1, x)
        return _mark("ffn.swiglu_fc2", M.comfy.ops.linear_input_act, self.fc2, h1, "swiglu")

    def adaln_fwd_fine(self, t_emb):
        return _mark("adaln", _orig_adaln, self, t_emb) if _prof["armed"] else adaln_fwd(self, t_emb)

    M.Attention.forward = attn_fwd_fine
    M.MLP.forward = mlp_fwd_fine
    M.AdalnProj.forward = adaln_fwd_fine

    # ---- total-forward timing + arming logic on _forward ----
    _orig_forward = M.MiniMaxH3Model._forward
    def forward_timed(self, *a, **k):
        idx = _state["call"]; _state["call"] += 1
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        deep = (idx == TARGET)
        if deep:
            _dump["nblocks"] = len(self.blocks); _dump["blk"] = 0
            _state["armed_dump"] = _dump["want"] and not _dump["done"]
            _prof["armed"] = True; _prof["ev"] = {}
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        s.record()
        out = _orig_forward(self, *a, **k)
        e.record(); torch.cuda.synchronize()
        total_ms = s.elapsed_time(e)
        with open(f"{LOGDIR}/profile_steps.jsonl", "a") as f:
            f.write(json.dumps({"idx": idx, "total_ms": round(total_ms, 3)}) + "\n")
        if deep:
            prof.__exit__(None, None, None)
            _prof["armed"] = False
            # sum component cuda ms
            comp = {}
            for name, pairs in _prof["ev"].items():
                comp[name] = round(sum(a.elapsed_time(b) for a, b in pairs), 3)
            summed = sum(comp.values())
            rep = {"target_idx": idx, "total_forward_ms": round(total_ms, 3),
                   "components_ms": comp, "sum_components_ms": round(summed, 3),
                   "other_ms": round(total_ms - summed, 3),
                   "components_pct": {k2: round(100 * v / total_ms, 1) for k2, v in comp.items()},
                   "num_blocks": len(self.blocks)}
            with open(f"{LOGDIR}/profile_step.json", "w") as f:
                json.dump(rep, f, indent=2)
            try:
                ka = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
                with open(f"{LOGDIR}/profile_trace.txt", "w") as f:
                    f.write(ka)
            except Exception as ex:
                with open(f"{LOGDIR}/profile_trace.txt", "w") as f:
                    f.write(f"key_averages failed: {ex}\n")
            print(f"[h3_profiler] deep profile step {idx} written. total={total_ms:.2f}ms comp={comp}", flush=True)
        return out
    M.MiniMaxH3Model._forward = forward_timed
    print(f"[h3_profiler] ARMED (target step={TARGET}). Wrapped H3 forward/attn/ffn/adaln/softmax.", flush=True)
