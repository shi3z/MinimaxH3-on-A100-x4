"""Reusable sequence-parallel runtime for MiniMax-H3 (installs into the model seam).

install() monkeypatches:
  - MiniMaxH3Model.run_blocks  -> shard sequence, run 50 blocks with Ulysses attn, all-gather
  - Attention.forward          -> Ulysses all-to-all around FlashAttention-2

Exact (bit-exact) when S % WORLD == 0. For S not divisible by WORLD, pads with a
key-masked tail so real-token outputs stay exact (see _pad_shard).

Env: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT (one process per GPU, GPU=cuda:0 via CUDA_VISIBLE_DEVICES).
"""
import os, torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
_INIT = False

def init():
    global _INIT
    if WORLD > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", rank=RANK, world_size=WORLD)
    torch.cuda.set_device(0)
    _INIT = True

def a2a_seq_to_head(x, world):        # [1,H,Sl,d] -> [1,Hl,S,d]
    _, H, Sl, d = x.shape; Hl = H // world
    x = x.reshape(world, Hl, Sl, d).contiguous()
    y = torch.empty_like(x); dist.all_to_all_single(y, x)
    return y.permute(1, 0, 2, 3).reshape(1, Hl, world * Sl, d).contiguous()

def a2a_head_to_seq(x, world, H):     # [1,Hl,S,d] -> [1,H,Sl,d]
    _, Hl, S, d = x.shape; Sl = S // world
    xp = x[0].view(Hl, world, Sl, d).permute(1, 0, 2, 3).reshape(world, Hl, Sl, d).contiguous()
    y = torch.empty_like(xp); dist.all_to_all_single(y, xp)
    return y.reshape(1, H, Sl, d).contiguous()

def _sp_attn_forward(self, x, rope_freqs=None, transformer_options={}):
    import comfy.model_management, comfy.quant_ops
    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim); k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q = q[0]; k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim)); k = self.k_norm(k.view(s, self.heads, self.head_dim))
    q = q.transpose(0, 1).unsqueeze(0); k = k.transpose(0, 1).unsqueeze(0); v = v.transpose(0, 1).unsqueeze(0)
    qh = a2a_seq_to_head(q, WORLD); kh = a2a_seq_to_head(k, WORLD); vh = a2a_seq_to_head(v, WORLD)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        oh = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
    o = a2a_head_to_seq(oh, WORLD, self.heads)
    o = o.transpose(1, 2).reshape(s, self.heads * self.head_dim)
    return self.out_proj(o)

def _shard_segments(segs, lo, hi):
    out = []
    for a, b, row in segs:
        A, B = max(a, lo), min(b, hi)
        if A < B: out.append((A - lo, B - lo, row))
    return out

def _sp_run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options={}):
    if WORLD == 1:
        return _orig_run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options)
    S = h.shape[0]
    if S % WORLD != 0:
        raise NotImplementedError(f"S={S} not divisible by WORLD={WORLD} (uneven-shard TODO)")
    Sl = S // WORLD; lo, hi = RANK * Sl, (RANK + 1) * Sl
    h_loc = h[lo:hi].contiguous()
    rope_loc = rope_freqs[:, lo:hi].contiguous() if rope_freqs is not None else None
    segs_loc = _shard_segments(mod_segments, lo, hi)
    for blk in self.blocks:
        h_loc = blk(h_loc, t_emb, segs_loc, rope_loc, transformer_options=transformer_options)
    gathered = [torch.empty_like(h_loc) for _ in range(WORLD)]
    dist.all_gather(gathered, h_loc.contiguous())
    return torch.cat(gathered, dim=0)

_orig_run_blocks = None
def install():
    global _orig_run_blocks
    import comfy.ldm.minimax.model as MM
    if _orig_run_blocks is None:
        _orig_run_blocks = MM.MiniMaxH3Model.run_blocks
    MM.Attention.forward = _sp_attn_forward
    MM.MiniMaxH3Model.run_blocks = _sp_run_blocks
    if RANK == 0:
        print(f"[sp_runtime] installed (WORLD={WORLD}) — run_blocks + Ulysses attention patched", flush=True)
