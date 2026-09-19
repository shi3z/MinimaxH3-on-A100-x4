"""MiniMax-H3 sequence-parallel DiT block-loop harness (GPU 4,5,6,7).

Verifies + benchmarks the 50-DiTBlock loop (the 95%+ hot region) with Ulysses
sequence parallelism against captured single-GPU ground truth.

Launch (one process per GPU, each sees its GPU as cuda:0):
  bench/run_sp.sh          # 4-GPU SP
  WORLD_SIZE unset -> single-process verify (Step B)

Ground truth: /mnt/ssdraid/project/h3-opt/logs/io_blockloop_{in,out}.pt
"""
import os, sys, time, json
sys.path.insert(0, "/mnt/ssdraid/project/comfy-h3")
import torch

LOGDIR = "/mnt/ssdraid/project/h3-opt/logs"
UNET = "/mnt/ssd/models/comfy-h3/diffusion_models/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
SP = WORLD > 1

def log(*a):
    if RANK == 0: print(*a, flush=True)

# ---- distributed init (one process per physical GPU; CUDA_VISIBLE_DEVICES=<gpu> => cuda:0) ----
if SP:
    import torch.distributed as dist
    dist.init_process_group("nccl", rank=RANK, world_size=WORLD)
DEV = "cuda:0"
torch.cuda.set_device(0)

# ---- load H3 diffusion model (same path as UNETLoader) ----
import comfy.sd, comfy.model_management as mm
log(f"[rank{RANK}] loading H3 model on {torch.cuda.get_device_name(0)} ...")
mp = comfy.sd.load_diffusion_model(UNET, model_options={})
mm.load_models_gpu([mp])
dm = mp.model.diffusion_model          # MiniMaxH3Model
blocks = dm.blocks
nblocks = len(blocks)
import comfy.ldm.minimax.model as MM
log(f"[rank{RANK}] loaded: {nblocks} blocks, hidden={dm.hidden_size}")

# ---- Ulysses all-to-all (validated in microbench) ----
def a2a_seq_to_head(x, world):
    _, Hh, Sl, d = x.shape; Hl = Hh // world
    x = x.reshape(world, Hl, Sl, d).contiguous()
    y = torch.empty_like(x); dist.all_to_all_single(y, x)
    return y.permute(1, 0, 2, 3).reshape(1, Hl, world * Sl, d).contiguous()
FP8 = os.environ.get("H3_SP_FP8") == "1"   # approximate: transport q/k/v in fp8 (half PCIe bytes)
def _a2a_bytes(x_u8):  # all-to-all on a [world, ...] uint8 tensor
    y = torch.empty_like(x_u8); dist.all_to_all_single(y, x_u8); return y
def a2a_seq_to_head_qkv(qkv, world):
    # qkv: [3, H, Sl, d] seq-sharded -> [3, Hl, S, d] head-sharded, in ONE all-to-all
    _, H, Sl, d = qkv.shape; Hl = H // world
    if FP8:
        amax = qkv.abs().amax(); dist.all_reduce(amax, op=dist.ReduceOp.MAX)
        scale = (448.0 / amax.clamp(min=1e-6)).to(torch.float32)
        q8 = (qkv.float() * scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        x = q8.view(3, world, Hl, Sl, d).permute(1, 0, 2, 3, 4).reshape(world, 3, Hl, Sl, d).contiguous().view(torch.uint8)
        y = _a2a_bytes(x).view(torch.float8_e4m3fn).view(world, 3, Hl, Sl, d)
        out = y.permute(1, 2, 0, 3, 4).reshape(3, Hl, world * Sl, d).contiguous()
        return (out.to(torch.float32) / scale).to(torch.bfloat16)
    x = qkv.view(3, world, Hl, Sl, d).permute(1, 0, 2, 3, 4).reshape(world, 3, Hl, Sl, d).contiguous()
    y = torch.empty_like(x); dist.all_to_all_single(y, x)
    return y.permute(1, 2, 0, 3, 4).reshape(3, Hl, world * Sl, d).contiguous()
def a2a_head_to_seq(x, world, H):
    _, Hl, S, d = x.shape; Sl = S // world
    xp = x[0].view(Hl, world, Sl, d).permute(1, 0, 2, 3).reshape(world, Hl, Sl, d).contiguous()
    if FP8:
        amax = xp.abs().amax(); dist.all_reduce(amax, op=dist.ReduceOp.MAX)
        scale = (448.0 / amax.clamp(min=1e-6)).to(torch.float32)
        x8 = (xp.float() * scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        y = _a2a_bytes(x8).view(torch.float8_e4m3fn).view(world, Hl, Sl, d)
        return (y.to(torch.float32) / scale).to(torch.bfloat16).reshape(1, H, Sl, d).contiguous()
    y = torch.empty_like(xp); dist.all_to_all_single(y, xp)
    return y.reshape(1, H, Sl, d).contiguous()

# ---- optional per-component profiling of the SP block loop ----
_SP = {"armed": False, "ev": {}}
def _spm(name, fn, *a, **k):
    if not _SP["armed"]:
        return fn(*a, **k)
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); r = fn(*a, **k); e.record()
    _SP["ev"].setdefault(name, []).append((s, e))
    return r

# ---- SP attention: mirrors model.Attention.forward, Ulysses around the softmax ----
from torch.nn.attention import SDPBackend, sdpa_kernel
def make_sp_attn(orig_cls):
    def sp_forward(self, x, rope_freqs=None, transformer_options={}):
        s = x.shape[0]
        q, k, v = _spm("attn.qkv", self.qkv_proj, x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            q = q.view(1, s, self.heads, self.head_dim); k = k.view(1, s, self.heads, self.head_dim)
            qw = MM.comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = MM.comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            MM.comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            q = q[0]; k = k[0]
        else:
            q = self.q_norm(q.view(s, self.heads, self.head_dim)); k = self.k_norm(k.view(s, self.heads, self.head_dim))
        q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)        # [H,Sl,d]
        # Ulysses: fused qkv seq-shard -> head-shard(full S) -> FA2 -> back
        qkv = torch.stack([q, k, v], dim=0).contiguous()                           # [3,H,Sl,d]
        fh = _spm("attn.a2a_qkv", a2a_seq_to_head_qkv, qkv, WORLD)                  # [3,Hl,S,d]
        qh, kh, vh = fh[0:1], fh[1:2], fh[2:3]                                      # each [1,Hl,S,d]
        def _sdpa():
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                return torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
        oh = _spm("attn.softmax", _sdpa)                                           # [1,Hl,S,d]
        o = _spm("attn.a2a_out", a2a_head_to_seq, oh, WORLD, self.heads)            # [1,H,Sl,d]
        o = o.transpose(1, 2).reshape(s, self.heads * self.head_dim)               # [Sl, inner]
        return _spm("attn.out", self.out_proj, o)
    return sp_forward

# ---- load ground truth ----
gi = torch.load(f"{LOGDIR}/io_blockloop_in.pt", map_location="cpu")
go = torch.load(f"{LOGDIR}/io_blockloop_out.pt", map_location="cpu")
h_in = gi["h_in"]; t_emb = gi["t_emb"].to(DEV); rope = gi["rope_freqs"]; mod_segments = gi["mod_segments"]
h_ref = go["h_out"]
S = h_in.shape[0]
assert S % WORLD == 0, f"S={S} not divisible by WORLD={WORLD}"
Sl = S // WORLD
lo, hi = RANK * Sl, (RANK + 1) * Sl

def shard_segments(segs, lo, hi):
    out = []
    for a, b, row in segs:
        A, B = max(a, lo), min(b, hi)
        if A < B: out.append((A - lo, B - lo, row))
    return out

def run_blocks(h, t, segs, rf):
    for blk in blocks:
        h = blk(h, t, segs, rf)
    return h

def bench(fn, iters=6, warmup=2):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    if SP: dist.barrier()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    t = st.elapsed_time(en) / iters
    if SP:
        tt = torch.tensor([t], device=DEV); dist.all_reduce(tt, op=dist.ReduceOp.MAX); t = tt.item()
    return t

if not SP:
    # Step B: single-GPU replay of all blocks on full h_in -> compare to h_out (weight parity)
    h = h_in.to(DEV); rf = rope.to(DEV)
    with torch.no_grad():
        out = run_blocks(h.clone(), t_emb, mod_segments, rf)
    err = (out.float() - h_ref.to(DEV).float()).abs()
    log(f"[verify Step B] max_err={err.max().item():.4e} mean_err={err.mean().item():.4e} "
        f"(ref absmax={h_ref.abs().max().item():.3f})")
    ms = bench(lambda: run_blocks(h.clone(), t_emb, mod_segments, rf))
    log(f"[bench Step B] 50-block loop single-GPU: {ms:.1f} ms")
else:
    # Step C: SP. patch attention, shard inputs, run, gather, verify + bench
    MM.Attention.forward = make_sp_attn(MM.Attention)
    _orig_mlp_fwd = MM.MLP.forward
    def _mlp_prof(self, x):
        h1 = _spm("ffn.fc1", self.fc1, x)
        return _spm("ffn.swiglu_fc2", MM.comfy.ops.linear_input_act, self.fc2, h1, "swiglu")
    MM.MLP.forward = _mlp_prof
    h_loc = h_in[lo:hi].to(DEV).contiguous()
    rf_loc = rope[:, lo:hi].to(DEV).contiguous()
    segs_loc = shard_segments(mod_segments, lo, hi)
    with torch.no_grad():
        out_loc = run_blocks(h_loc.clone(), t_emb, segs_loc, rf_loc)
    # all_gather (NCCL supports all_gather, not gather)
    gathered = [torch.empty_like(out_loc) for _ in range(WORLD)]
    dist.all_gather(gathered, out_loc.contiguous())
    if RANK == 0:
        out_full = torch.cat(gathered, dim=0)
        err = (out_full.float() - h_ref.to(DEV).float()).abs()
        log(f"[verify Step C SP{WORLD}] max_err={err.max().item():.4e} mean_err={err.mean().item():.4e} "
            f"(ref absmax={h_ref.abs().max().item():.3f})")
    # one-shot component profile
    torch.cuda.synchronize(); dist.barrier()
    _SP["armed"] = True; _SP["ev"] = {}
    tot_s = torch.cuda.Event(enable_timing=True); tot_e = torch.cuda.Event(enable_timing=True)
    tot_s.record(); _ = run_blocks(h_loc.clone(), t_emb, segs_loc, rf_loc); tot_e.record()
    torch.cuda.synchronize(); _SP["armed"] = False
    comp = {n: round(sum(a.elapsed_time(b) for a, b in ev), 1) for n, ev in _SP["ev"].items()}
    tot = tot_s.elapsed_time(tot_e); comp_sum = sum(comp.values())
    if RANK == 0:
        log(f"[SP components rank0] total={tot:.0f}ms sum={comp_sum:.0f}ms other={tot-comp_sum:.0f}ms")
        for n in sorted(comp, key=lambda x: -comp[x]):
            log(f"    {n:16s} {comp[n]:8.1f} ms  {100*comp[n]/tot:5.1f}%")
    ms = bench(lambda: run_blocks(h_loc.clone(), t_emb, segs_loc, rf_loc))
    log(f"[bench Step C SP{WORLD}] 50-block loop: {ms:.1f} ms/step  (single-GPU baseline was ~18500 ms)")
    dist.destroy_process_group()
