"""Persistent 4-rank Ulysses-SP H3 inference SERVICE (GPU4-7).

Loads the H3 stack ONCE, then serves many requests in a loop (models never reloaded per
request -> model-load is amortized to zero, which is the real full-gen latency win).

Execution model: 4-rank SPMD. Every rank runs the IDENTICAL cut graph with the SAME seed, so
run_blocks reaches the Ulysses collectives in lockstep (sp_runtime patches DiT run_blocks +
attention). This SPMD-identical-graph invariant is exactly what makes the output bit-identical
to the 1-GPU reference; we keep it. rank 0's SaveVideo is the deliverable.

Per-request stage budget is measured with monkeypatch timers:
  request-parse | model-load | text-encode | ref-VAE-encode | DiT (SP) | VAE-decode | mux/save

Optional SP_RANK0_ENCODE=1: text-encode runs only on rank 0 and is broadcast to ranks 1-3
(compute-skip; keeps CLIP loaded on all ranks so it is neutral on VRAM and on wall-clock —
reported with numbers).

Modes:
  manifest:  python sp_service.py <manifest.json>     # [{"job": "<path>", "tag": "<t>"}, ...] run in order
  serve:     python sp_service.py --serve <queue_dir> # rank0 polls dir, broadcasts job to all ranks, loops
Launch all 4 ranks with run_sp_service.sh.
"""
import os, sys, time, json, glob, statistics
_PROC_T0 = time.time()   # process start (for startup = bootstrap + NCCL + install, reported once)
sys.path.insert(0, "/mnt/ssdraid/project/h3-opt/harness")
sys.path.insert(0, "/mnt/ssdraid/project/comfy-h3")
sys.path.insert(0, "/mnt/ssdraid/project/h3-fleet")

LOG = "/mnt/ssdraid/project/h3-opt/logs"
COMFY_OUT = "/mnt/ssdraid/project/comfy-h3/output"
RANK0_ENCODE = os.environ.get("SP_RANK0_ENCODE") == "1"

import sp_runtime
sp_runtime.init()
RANK, WORLD = sp_runtime.RANK, sp_runtime.WORLD
import torch
import torch.distributed as dist

def log(*a):
    if RANK == 0:
        print("[svc]", *a, flush=True)

# ---- ComfyUI bootstrap (ONCE) --------------------------------------------------
os.chdir("/mnt/ssdraid/project/comfy-h3")
import asyncio
import server, execution, nodes, folder_paths
import utils.extra_config
_cfg = os.path.join(os.getcwd(), "extra_model_paths.yaml")
if os.path.isfile(_cfg):
    utils.extra_config.load_extra_path_config(_cfg)
loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
ps = server.PromptServer(loop)
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
import comfy.model_management as mm
import comfy.sd
sp_runtime.install()
BOOTSTRAP_S = time.time() - _PROC_T0   # import + NCCL init + comfy node init + SP install (model load is lazy, measured as 'model_load' in the first request)
log(f"startup: bootstrap+NCCL+install = {BOOTSTRAP_S:.1f}s (model/LoRA/VAE/TE load is lazy -> counted as model_load in the warmup request)")

# ---- per-request stage timers (wall-clock ms; CUDA-synced) ----------------------
STAGE = {}
def _acc(name, ms):
    STAGE[name] = STAGE.get(name, 0.0) + ms
class _wt:
    def __init__(self, name): self.name = name
    def __enter__(self):
        torch.cuda.synchronize(); self.t = time.time(); return self
    def __exit__(self, *a):
        torch.cuda.synchronize(); _acc(self.name, (time.time() - self.t) * 1000.0); return False

# model load (loader nodes)
for _cls, _fn, _stg in [("UNETLoader", None, "model_load"), ("CLIPLoader", None, "model_load"),
                        ("VAELoader", None, "model_load")]:
    C = nodes.NODE_CLASS_MAPPINGS.get(_cls)
    if C and getattr(C, "FUNCTION", None):
        _orig = getattr(C, C.FUNCTION)
        def _mk(orig, stg):
            def w(self, *a, **k):
                with _wt(stg): return orig(self, *a, **k)
            return w
        setattr(C, C.FUNCTION, _mk(_orig, "model_load"))
# LoRA loader
_LC = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3TurboLoRA")
if _LC and getattr(_LC, "FUNCTION", None):
    _lo = getattr(_LC, _LC.FUNCTION)
    def _lw(self, *a, **k):
        with _wt("model_load"): return _lo(self, *a, **k)
    setattr(_LC, _LC.FUNCTION, _lw)

# text encode  (comfy.sd.CLIP.encode_from_tokens_scheduled) + optional rank0-only broadcast
_orig_enc = comfy.sd.CLIP.encode_from_tokens_scheduled
def _enc(self, *a, **k):
    with _wt("text_encode"):
        if RANK0_ENCODE and WORLD > 1:
            out = _orig_enc(self, *a, **k) if RANK == 0 else None
            box = [out]; dist.broadcast_object_list(box, src=0)
            return box[0]
        return _orig_enc(self, *a, **k)
comfy.sd.CLIP.encode_from_tokens_scheduled = _enc

# ref VAE encode + VAE decode  (comfy.sd.VAE.* — API-agnostic; VAEDecode/VAEDecodeAudio v3 nodes
# call these internally, and their FUNCTION cannot be setattr-patched without breaking the v3 adapter)
_orig_venc = comfy.sd.VAE.encode
def _venc(self, *a, **k):
    with _wt("ref_vae_encode"): return _orig_venc(self, *a, **k)
comfy.sd.VAE.encode = _venc
_orig_vdec = comfy.sd.VAE.decode
def _vdec(self, *a, **k):
    with _wt("vae_decode"): return _orig_vdec(self, *a, **k)
comfy.sd.VAE.decode = _vdec

# DiT (SP run_blocks)
import comfy.ldm.minimax.model as MM
_rb = MM.MiniMaxH3Model.run_blocks
_dit_calls = []
def _trb(self, h, *a, **k):
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record(); out = _rb(self, h, *a, **k); en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en); _dit_calls.append(ms); _acc("dit_sp", ms); return out
MM.MiniMaxH3Model.run_blocks = _trb

# ---- mux/save: replace ComfyUI's per-frame PyAV loop with ONE streaming ffmpeg (rawvideo stdin
#      -> libx264), or an instrumented copy of the original for profiling. Accumulates "mux_save".
import comfy_api.latest._input_impl.video_types as VT
import numpy as np, subprocess, tempfile, hashlib, math as _math
from fractions import Fraction
STREAM_SAVE = os.environ.get("SP_STREAM_SAVE", "1") == "1"
HASH_FRAMES = os.environ.get("SP_HASH_FRAMES") == "1"

def _frames_to_u8(images):
    # [T,H,W,3] float 0..1 (GPU/CPU) -> contiguous CPU uint8 RGB
    return (images.clamp(0, 1) * 255).round().byte().cpu().contiguous().numpy()

def _stream_save(self, path, format=VT.VideoContainer.AUTO, codec=VT.VideoCodec.AUTO,
                 metadata=None, bit_depth=None, crf=None):
    comp = self._VideoFromComponents__components
    t0 = time.time()
    arr = _frames_to_u8(comp.images)                    # tensor -> cpu uint8
    T, H, W, _ = arr.shape
    fps = float(comp.frame_rate)
    if HASH_FRAMES and RANK == 0:
        open(f"{LOG}/frames_hash_{os.path.basename(path)}.txt", "w").write(
            hashlib.md5(arr.tobytes()).hexdigest())
    araw = None; ach = 0; asr = 0
    if comp.audio:
        asr = int(comp.audio["sample_rate"])
        wf = comp.audio["waveform"][0]                  # [C, L]
        n = _math.ceil((asr / fps) * T)
        wf = wf[:, :n].float().cpu().contiguous()
        ach = wf.shape[0]
        # raw f32le interleaved PCM (no torchaudio/torchcodec dependency), fed as a 2nd ffmpeg input
        inter = wf.transpose(0, 1).contiguous().numpy().astype(np.float32)   # [L, C]
        araw = tempfile.NamedTemporaryFile(suffix=".pcm", delete=False).name
        with open(araw, "wb") as f: f.write(inter.tobytes())
    cq = str(int(crf)) if crf is not None else "16"
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-"]
    if araw: cmd += ["-f", "f32le", "-ar", str(asr), "-ac", str(ach), "-i", araw]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", cq, "-pix_fmt", "yuv420p"]
    if araw: cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(arr.tobytes()); p.stdin.close(); rc = p.wait()
    if araw:
        try: os.remove(araw)
        except OSError: pass
    if rc != 0:
        raise RuntimeError(f"ffmpeg streaming encode failed rc={rc}")
    _acc("mux_save", (time.time() - t0) * 1000.0)

def _instrumented_orig_save(self, path, format=VT.VideoContainer.AUTO, codec=VT.VideoCodec.AUTO,
                            metadata=None, bit_depth=None, crf=None):
    import av
    comp = self._VideoFromComponents__components
    t0 = time.time()
    if HASH_FRAMES and RANK == 0:
        open(f"{LOG}/frames_hash_{os.path.basename(path)}.txt", "w").write(
            hashlib.md5(_frames_to_u8(comp.images).tobytes()).hexdigest())
    open_kwargs = VT.mp4_output_open_kwargs(path, format, codec)
    with av.open(path, **open_kwargs) as output:
        fr = Fraction(round(comp.frame_rate * 1000), 1000); pix = "yuv420p"
        vs = output.add_stream("h264", rate=fr); vs.width = comp.images.shape[2]; vs.height = comp.images.shape[1]; vs.pix_fmt = pix
        if crf is not None: vs.options = {"crf": str(crf)}
        astream = None
        if comp.audio:
            asr = int(comp.audio["sample_rate"]); wf = comp.audio["waveform"]
            wf = wf[0, :, :_math.ceil((asr / fr) * comp.images.shape[0])]
            layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(wf.shape[0], "stereo")
            astream = output.add_stream("aac", rate=asr, layout=layout)
        for frame in comp.images:
            img = (frame * 255).clamp(0, 255).byte().cpu().numpy()
            vf = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format=pix)
            output.mux(vs.encode(vf))
        output.mux(vs.encode(None))
        if astream and comp.audio:
            af = av.AudioFrame.from_ndarray(wf.float().cpu().contiguous().numpy(), format="fltp", layout=layout)
            af.sample_rate = asr; af.pts = 0
            output.mux(astream.encode(af)); output.mux(astream.encode(None))
    _acc("mux_save", (time.time() - t0) * 1000.0)

VT.VideoFromComponents.save_to = _stream_save if STREAM_SAVE else _instrumented_orig_save

# ---- distributed VIDEO VAE decode across ranks (bit-exact) ---------------------------------
# The H3 video VAE decodes the latent in INDEPENDENT temporal chunks (tokens_chunk_size=5,
# token_overlap=2 -> 7-token clips); each _adaptive_decode(clip_z) depends only on its clip and
# the chunks are stitched afterward by deterministic blend. The final latent is identical on all
# ranks (SPMD sampler), so each rank decodes a subset of chunks; the raw per-chunk pixels are
# NCCL-broadcast to all ranks; every rank then runs the ORIGINAL stitch code unchanged -> the
# result is bit-identical to the single-GPU decode. (Audio VAE decode is 0.18s -> left replicated.)
DIST_VAE = os.environ.get("SP_DIST_VAE", "1") == "1"
import comfy.ldm.minimax.vae as VAEMOD
_orig_decode_temporal = VAEMOD.MiniMaxH3VideoVAE.decode_temporal
_DT = {torch.float16: 0, torch.float32: 1, torch.bfloat16: 2}
_DT_INV = {v: k for k, v in _DT.items()}

def _dist_decode_temporal(self, z):
    if WORLD == 1 or not dist.is_initialized():
        return _orig_decode_temporal(self, z)
    cs = self.tokens_chunk_size; ov = self.token_overlap
    chunk_dec = cs * self.vae_ratio_t
    split_count = int(self.token_drop > 0) + 1
    pseudo = z.shape[2] + self.token_drop
    pad_tokens = 0
    rem = pseudo % cs
    if rem != 0:
        pad_tokens = cs - rem; pseudo += pad_tokens
    num_chunks = pseudo // cs - int(self.token_drop > 0)
    if num_chunks < 1:
        pad_tokens += cs; num_chunks += 1
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    output_frames = self._decode_temporal_frame_plan(z.shape[2], num_chunks, pad_tokens)

    W = min(WORLD, num_chunks)
    # phase 1: each rank decodes its OWNED chunks first — runs in PARALLEL across ranks, no
    # collectives here (decoupling compute from the broadcast is essential: interleaving them
    # serializes the ranks because everyone blocks on each owner's broadcast).
    local = {}
    for i in range(num_chunks):
        if i % W == RANK:
            t0 = i * cs; t1 = t0 + cs + ov
            local[i] = self._adaptive_decode(z[:, :, t0:t1, :, :]).contiguous()
    # phase 2: broadcast each chunk's pixels from its owner (owner already computed it in phase 1)
    all_clip = [None] * num_chunks
    for i in range(num_chunks):
        owner = i % W
        cd = local.get(i)
        if owner == RANK:
            meta = torch.tensor(list(cd.shape) + [_DT[cd.dtype]], device=z.device, dtype=torch.long)
        else:
            meta = torch.empty(6, device=z.device, dtype=torch.long)
        dist.broadcast(meta, src=owner)
        if owner != RANK:
            cd = torch.empty(tuple(meta[:5].tolist()), device=z.device, dtype=_DT_INV[int(meta[5].item())])
        dist.broadcast(cd, src=owner)
        all_clip[i] = cd

    # --- stitch: ORIGINAL logic verbatim, using gathered per-chunk pixels (bit-exact) ---
    dec = None; dec_overlap = None; write_pos = 0
    def write_part(part):
        nonlocal dec, write_pos
        pf = part.shape[2]
        if pf <= 0: return
        if dec is None:
            osh = list(part.shape); osh[2] = output_frames
            dec = torch.empty(osh, dtype=part.dtype, device=part.device)
        cf = min(pf, max(0, dec.shape[2] - write_pos))
        if cf > 0:
            dec[:, :, write_pos:write_pos + cf, :, :].copy_(part[:, :, :cf, :, :]); write_pos += cf
    for i in range(num_chunks):
        clip_dec = all_clip[i]
        for j in range(split_count):
            fs = j * chunk_dec; fe = min(fs + chunk_dec, clip_dec.shape[2])
            cchunk = clip_dec[:, :, fs:fe, :, :][:, :, self.frame_pre_padding:, :, :]
            if j == 0:
                if dec_overlap is not None:
                    cchunk = self.blend(dec_overlap, cchunk, self.frame_overlap, dim=-3); dec_overlap = None
                write_part(cchunk)
            else:
                dec_overlap = cchunk.contiguous()
        if i == num_chunks - 1 and dec_overlap is not None:
            write_part(dec_overlap); dec_overlap = None
    return dec

if DIST_VAE:
    VAEMOD.MiniMaxH3VideoVAE.decode_temporal = _dist_decode_temporal

# ---- persistent executor (RAM_PRESSURE cache keeps loaded models across requests) ----
from comfy_worker import build_graph
cache_ram = min(10.0, max(2.0, mm.total_ram * 0.10 / 1024.0))
cache_ram_inactive = min(128.0, mm.total_ram / 1024.0)
E = execution.PromptExecutor(ps, cache_type=execution.CacheType.RAM_PRESSURE,
                             cache_args={"lru": 0, "ram": cache_ram, "ram_inactive": cache_ram_inactive})

def handle(job, tag):
    STAGE.clear(); _dit_calls.clear()
    tpar = time.time()
    g = build_graph(job)
    for nid, node in g.items():
        if node.get("class_type") == "SaveVideo":
            node["inputs"]["filename_prefix"] = f"spgen/svc_{tag}_r{RANK}"
    _acc("request_parse", (time.time() - tpar) * 1000.0)
    prompt_id = f"svc_{tag}_r{RANK}_{int(time.time()*1000)}"
    valid = loop.run_until_complete(execution.validate_prompt(prompt_id, g, None))
    if not valid[0]:
        log("VALIDATION FAILED:", valid[1]); return None
    if WORLD > 1: dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    E.execute(g, prompt_id, extra_data={}, execute_outputs=valid[2])
    if WORLD > 1: dist.barrier()
    dt = (time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    if WORLD > 1:
        pm = torch.tensor([peak], device="cuda:0"); dist.all_reduce(pm, op=dist.ReduceOp.MAX); peak = pm.item()
    mp4 = None
    c = sorted(glob.glob(f"{COMFY_OUT}/spgen/svc_{tag}_r0*.mp4"), key=os.path.getmtime, reverse=True)
    if c: mp4 = c[0]
    if RANK == 0:
        budget = {k: round(v, 1) for k, v in STAGE.items()}
        dit_med = round(statistics.median(_dit_calls[1:]) if len(_dit_calls) > 2 else (statistics.median(_dit_calls) if _dit_calls else 0), 1)
        res = {"tag": tag, "world": WORLD, "seed": job.get("seed"), "rank0_encode": RANK0_ENCODE,
               "wall_s": round(dt, 2), "peak_mem_gb": round(peak, 2),
               "dit_calls": len(_dit_calls), "dit_ms_median": dit_med,
               "stage_ms": budget, "mp4": mp4}
        overhead = round(dt * 1000 - sum(STAGE.values()), 1)  # barriers/validate/model-mgmt/create-video wrap
        res["stage_ms"]["other_sync"] = overhead
        json.dump(res, open(f"{LOG}/svc_{tag}_result.json", "w"), indent=1)
        log(f"=== {tag} === wall={dt:.2f}s peak={peak:.1f}GB mp4={mp4}")
        for k in ["request_parse","model_load","text_encode","ref_vae_encode","dit_sp","vae_decode","mux_save"]:
            if k in budget: log(f"    {k:16s} {budget[k]/1000:7.2f} s")
        log(f"    other_sync        {overhead/1000:7.2f} s")
        log(f"    dit steps={len(_dit_calls)} median={dit_med} ms  all={[round(x) for x in _dit_calls]}")
    return mp4

# ---- request source ------------------------------------------------------------
def _bcast_job(job):
    if WORLD > 1:
        box = [job]; dist.broadcast_object_list(box, src=0); return box[0]
    return job

if len(sys.argv) >= 3 and sys.argv[1] == "--serve":
    QDIR = sys.argv[2]
    log(f"SERVICE up (persistent, WORLD={WORLD}, rank0_encode={RANK0_ENCODE}) polling {QDIR}")
    seen = 0
    while True:
        job = None; tag = None
        if RANK == 0:
            paths = sorted(glob.glob(f"{QDIR}/*.json"))
            if paths:
                p = paths[0]
                try:
                    job = json.load(open(p)); job = job.get("job", job)
                    tag = f"cut{job.get('cut','x')}_{seen}"
                    os.rename(p, p + ".taken")
                except Exception as ex:
                    log("bad job", p, ex); job = None
        flag = [1 if job is not None else 0]
        if WORLD > 1: dist.broadcast_object_list(flag, src=0)
        if not flag[0]:
            time.sleep(2); continue
        job = _bcast_job(job); tag = _bcast_job(tag)
        handle(job, tag); seen += 1
else:
    MAN = sys.argv[1]
    manifest = json.load(open(MAN))
    log(f"MANIFEST run: {len(manifest)} requests (persistent, WORLD={WORLD}, rank0_encode={RANK0_ENCODE})")
    for i, req in enumerate(manifest):
        job = json.load(open(req["job"]))
        tag = req.get("tag", f"req{i}")
        handle(job, tag)
    if WORLD > 1:
        dist.barrier(); dist.destroy_process_group()
