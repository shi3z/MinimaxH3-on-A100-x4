"""Isolated 4-rank test of distributed H3 video-VAE decode (GPU4-7).
Times single-GPU vs distributed decode of a fixed synthetic latent, verifies bit-exactness."""
import os, sys, time, torch
sys.path.insert(0, "/mnt/ssdraid/project/h3-opt/harness")
sys.path.insert(0, "/mnt/ssdraid/project/comfy-h3")
os.chdir("/mnt/ssdraid/project/comfy-h3")
import sp_runtime
sp_runtime.init()
RANK, WORLD = sp_runtime.RANK, sp_runtime.WORLD
import torch.distributed as dist
import asyncio, server, nodes, utils.extra_config
utils.extra_config.load_extra_path_config("extra_model_paths.yaml")
loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
ps = server.PromptServer(loop)
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
import comfy.model_management as mm
import comfy.ldm.minimax.vae as VAEMOD

VL = nodes.NODE_CLASS_MAPPINGS["VAELoader"]()
vae = VL.load_vae("minimax_h3_video_vae_fp16.safetensors")[0]
mm.load_models_gpu([vae.patcher])
m = vae.first_stage_model
_orig = VAEMOD.MiniMaxH3VideoVAE.decode_temporal
_DT = {torch.float16: 0, torch.float32: 1, torch.bfloat16: 2}; _DTI = {v: k for k, v in _DT.items()}
_hit = [0]

def dist_decode_temporal(self, z, active_world):
    cs = self.tokens_chunk_size; ov = self.token_overlap
    chunk_dec = cs * self.vae_ratio_t; split_count = int(self.token_drop > 0) + 1
    pseudo = z.shape[2] + self.token_drop; pad = 0; rem = pseudo % cs
    if rem: pad = cs - rem; pseudo += pad
    num_chunks = pseudo // cs - int(self.token_drop > 0)
    if num_chunks < 1: pad += cs; num_chunks += 1
    if pad > 0: z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)
    output_frames = self._decode_temporal_frame_plan(z.shape[2], num_chunks, pad)
    W = min(active_world, num_chunks)
    # phase 1: parallel per-rank decode of owned chunks (no collectives)
    local = {}; owned = 0
    for i in range(num_chunks):
        if i % W == RANK:
            t0 = i * cs; t1 = t0 + cs + ov
            local[i] = self._adaptive_decode(z[:, :, t0:t1]).contiguous(); owned += 1
    _hit[0] = owned
    # phase 2: broadcast pixels from each owner
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
            cd = torch.empty(tuple(meta[:5].tolist()), device=z.device, dtype=_DTI[int(meta[5])])
        dist.broadcast(cd, src=owner); all_clip[i] = cd
    dec = None; dov = None; wp = 0
    def wr(part):
        nonlocal dec, wp
        pf = part.shape[2]
        if pf <= 0: return
        if dec is None:
            osh = list(part.shape); osh[2] = output_frames; dec = torch.empty(osh, dtype=part.dtype, device=part.device)
        cf = min(pf, max(0, dec.shape[2] - wp))
        if cf > 0: dec[:, :, wp:wp + cf].copy_(part[:, :, :cf]); wp += cf
    for i in range(num_chunks):
        clip_dec = all_clip[i]
        for j in range(split_count):
            fs = j * chunk_dec; fe = min(fs + chunk_dec, clip_dec.shape[2])
            cc = clip_dec[:, :, fs:fe][:, :, self.frame_pre_padding:]
            if j == 0:
                if dov is not None: cc = self.blend(dov, cc, self.frame_overlap, dim=-3); dov = None
                wr(cc)
            else: dov = cc.contiguous()
        if i == num_chunks - 1 and dov is not None: wr(dov); dov = None
    return dec

torch.manual_seed(1234)
z = torch.randn(1, 24, 62, 45, 80, device="cuda:0", dtype=torch.float16)

def bench_local(fn, n=2):   # rank0-only, NO collectives
    with torch.no_grad(): out = fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        with torch.no_grad(): out = fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n, out

def bench_dist(fn, n=2):     # all ranks, with barriers
    with torch.no_grad(): fn()
    torch.cuda.synchronize(); dist.barrier()
    t = time.time()
    for _ in range(n):
        with torch.no_grad(): out = fn()
    torch.cuda.synchronize(); dist.barrier()
    return (time.time() - t) / n, out

# single-GPU reference: rank0 times original locally; broadcast ref tensor for max_err checks.
if RANK == 0:
    t1, ref = bench_local(lambda: _orig(m, z.clone()))
    print(f"[1-GPU] video decode {t1:.2f}s out {tuple(ref.shape)} {ref.dtype}", flush=True)
    meta = torch.tensor(list(ref.shape) + [_DT[ref.dtype]], device="cuda:0", dtype=torch.long)
else:
    meta = torch.empty(6, device="cuda:0", dtype=torch.long)
dist.broadcast(meta, src=0)
if RANK != 0:
    ref = torch.empty(tuple(meta[:5].tolist()), device="cuda:0", dtype=_DTI[int(meta[5])])
dist.broadcast(ref, src=0)
dist.barrier()

for AW in ([2, 4] if WORLD >= 4 else [WORLD]):
    tt, out = bench_dist(lambda: dist_decode_temporal(m, z.clone(), AW))
    if RANK == 0:
        err = (out.float() - ref.float()).abs().max().item()
        print(f"[{AW}-GPU tiled] video decode {tt:.2f}s  owned_chunks(rank0)={_hit[0]}  max_err_vs_1GPU={err:.3e}", flush=True)
    dist.barrier()
dist.destroy_process_group()
