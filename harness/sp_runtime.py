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

# SP attention is scoped to the DiT block loop ONLY. The same Attention class is also
# used by the TokenRefiner (full, replicated sequence, outside run_blocks) — running
# Ulysses there would all-to-all an unsharded tensor and corrupt conditioning. run_blocks
# sets _SP_ACTIVE while the sharded block loop runs; the patched attention falls back to
# the original math otherwise. _SP_REAL_S is the true (unpadded) sequence length: after the
# Ulysses gather each rank holds S_pad keys; slicing K,V to the real length drops the pad
# tail so real-token queries attend over exactly the real keys (bit-exact, no attn mask).
_SP_ACTIVE = False
_SP_REAL_S = None
_orig_attn_forward = None

# ---- optional per-op timing (SP_TIMING=1): CUDA-event category accumulators, summed per run_blocks ----
_TIMING = os.environ.get("SP_TIMING") == "1"
_acc = {}          # category -> list of (start_evt, end_evt)
_last = {}         # category -> summed ms for the most recent run_blocks call (rank-local)

class _tr:
    def __init__(self, cat): self.cat = cat
    def __enter__(self):
        if _TIMING:
            self.s = torch.cuda.Event(enable_timing=True); self.e = torch.cuda.Event(enable_timing=True)
            self.s.record()
        return self
    def __exit__(self, *a):
        if _TIMING:
            self.e.record(); _acc.setdefault(self.cat, []).append((self.s, self.e))
        return False

def _finalize_timing():
    global _acc
    if not _TIMING: return
    torch.cuda.synchronize()
    _last.clear()
    for cat, pairs in _acc.items():
        _last[cat] = sum(s.elapsed_time(e) for s, e in pairs)
    _acc = {}

def get_timing():
    return dict(_last)

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
    if not _SP_ACTIVE:
        return _orig_attn_forward(self, x, rope_freqs=rope_freqs, transformer_options=transformer_options)
    import comfy.model_management, comfy.quant_ops
    s = x.shape[0]
    with _tr("qkv"):
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
    with _tr("a2a_fwd"):
        qh = a2a_seq_to_head(q, WORLD); kh = a2a_seq_to_head(k, WORLD); vh = a2a_seq_to_head(v, WORLD)
    with _tr("fa2_local"):
        # drop the padded key/value tail so real-token queries attend over exactly the real
        # keys (unequal-length flash: q=S_pad, k/v=S_real -> real-query rows bit-exact).
        if _SP_REAL_S is not None and _SP_REAL_S < kh.shape[2]:
            kh = kh[:, :, :_SP_REAL_S, :]; vh = vh[:, :, :_SP_REAL_S, :]
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            oh = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
    with _tr("a2a_inv"):
        o = a2a_head_to_seq(oh, WORLD, self.heads)
    with _tr("out_proj"):
        o = o.transpose(1, 2).reshape(s, self.heads * self.head_dim)
        return self.out_proj(o)

# ---- A-1: comm/compute overlap (exact) ----------------------------------------
# K,V are gathered once (must precede any FA-2). The Q input all-to-all and the O
# output all-to-all are split into P pieces along the local-token axis and software-
# pipelined on a side stream, so piece p+1's transfers hide under piece p's
# FlashAttention. Math is byte-identical to _sp_attn_forward -> bit-exact.
_PIECES = int(os.environ.get("SP_PIECES", "4"))
_comm_stream = None

def _piece_bounds(n, p):
    base, rem = divmod(n, p)
    out, s = [], 0
    for i in range(p):
        ln = base + (1 if i < rem else 0)
        if ln == 0:
            continue
        out.append((s, s + ln)); s += ln
    return out

def _sp_attn_forward_overlap(self, x, rope_freqs=None, transformer_options={}):
    if not _SP_ACTIVE:
        return _orig_attn_forward(self, x, rope_freqs=rope_freqs, transformer_options=transformer_options)
    import comfy.model_management, comfy.quant_ops
    global _comm_stream
    if _comm_stream is None:
        _comm_stream = torch.cuda.Stream()
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
    # gather full K,V (blocking) — required before any attention
    kh = a2a_seq_to_head(k, WORLD); vh = a2a_seq_to_head(v, WORLD)
    if _SP_REAL_S is not None and _SP_REAL_S < kh.shape[2]:
        kh = kh[:, :, :_SP_REAL_S, :]; vh = vh[:, :, :_SP_REAL_S, :]
    Hl = self.heads // WORLD
    pieces = _piece_bounds(s, _PIECES)
    default = torch.cuda.current_stream()
    _comm_stream.wait_stream(default)                       # q,kh,vh ready before comm uses them
    o = torch.empty(1, self.heads, s, self.head_dim, device=x.device, dtype=q.dtype)
    # prime: gather Q piece 0 on comm stream
    with torch.cuda.stream(_comm_stream):
        qh_next = a2a_seq_to_head(q[:, :, pieces[0][0]:pieces[0][1], :].contiguous(), WORLD)
    for i, (lo, hi) in enumerate(pieces):
        default.wait_stream(_comm_stream)                  # qh for this piece is gathered
        qh_p = qh_next
        if i + 1 < len(pieces):                            # overlap: gather next Q piece while we compute
            nlo, nhi = pieces[i + 1]
            with torch.cuda.stream(_comm_stream):
                qh_next = a2a_seq_to_head(q[:, :, nlo:nhi, :].contiguous(), WORLD)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):    # FA-2 on default stream, hides comm above
            oh_p = torch.nn.functional.scaled_dot_product_attention(qh_p, kh, vh)
        with torch.cuda.stream(_comm_stream):              # scatter O piece back, overlaps next FA-2
            _comm_stream.wait_stream(default)
            o[:, :, lo:hi, :] = a2a_head_to_seq(oh_p, WORLD, self.heads)
    default.wait_stream(_comm_stream)                      # all O scatters done
    o = o.transpose(1, 2).reshape(s, self.heads * self.head_dim)
    return self.out_proj(o)

def _shard_segments(segs, lo, hi):
    out = []
    for a, b, row in segs:
        A, B = max(a, lo), min(b, hi)
        if A < B: out.append((A - lo, B - lo, row))
    return out

def _sp_block_timed(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
    # timing-instrumented replica of DiTBlock.forward: separates adaln/norm/residual from ffn,
    # and lets the attention (self.attn, patched) time its own sub-ops. Only used when SP_TIMING=1.
    import comfy.ldm.minimax.model as MM
    with _tr("norm_mod"):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = MM._mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
    a = self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
    with _tr("residual"):
        x = MM._mod_gate(x, gate_msa, a, mod_segments)
    with _tr("norm_mod"):
        h = MM._mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
    with _tr("ffn"):
        m = self.mlp(h)
    with _tr("residual"):
        return MM._mod_gate(x, gate_mlp, m, mod_segments)

def _sp_run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options={}):
    global _SP_ACTIVE, _SP_REAL_S
    if WORLD == 1:
        return _orig_run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options)
    S = h.shape[0]
    # EXACT uneven-S padding: pad the packed sequence up to a multiple of WORLD. Pad rows are
    # appended at the tail (outside every mod segment, so scale/shift/gate skip them), sharded
    # like real rows, run through the 50 blocks, then dropped after the all-gather. Their KEY
    # positions are excluded inside attention (via _SP_REAL_S) so real tokens never see pad.
    S_pad = ((S + WORLD - 1) // WORLD) * WORLD
    P = S_pad - S
    if P > 0:
        h = torch.cat([h, h.new_zeros(P, h.shape[1])], dim=0)
        if rope_freqs is not None:
            pad_shape = (rope_freqs.shape[0], P) + tuple(rope_freqs.shape[2:])
            rope_freqs = torch.cat([rope_freqs, rope_freqs.new_zeros(pad_shape)], dim=1)
    Sl = S_pad // WORLD; lo, hi = RANK * Sl, (RANK + 1) * Sl
    h_loc = h[lo:hi].contiguous()
    rope_loc = rope_freqs[:, lo:hi].contiguous() if rope_freqs is not None else None
    segs_loc = _shard_segments(mod_segments, lo, hi)   # pad rows (>= S) fall in no segment
    blkfn = _sp_block_timed if _TIMING else None
    _SP_ACTIVE = True; _SP_REAL_S = S
    try:
        for blk in self.blocks:
            if blkfn is not None:
                h_loc = blkfn(blk, h_loc, t_emb, segs_loc, rope_loc, transformer_options=transformer_options)
            else:
                h_loc = blk(h_loc, t_emb, segs_loc, rope_loc, transformer_options=transformer_options)
        with _tr("gather"):
            gathered = [torch.empty_like(h_loc) for _ in range(WORLD)]
            dist.all_gather(gathered, h_loc.contiguous())
            out = torch.cat(gathered, dim=0)
    finally:
        _SP_ACTIVE = False; _SP_REAL_S = None
    _finalize_timing()
    return out[:S] if P > 0 else out

_orig_run_blocks = None
def install():
    global _orig_run_blocks, _orig_attn_forward
    import comfy.ldm.minimax.model as MM
    if _orig_run_blocks is None:
        _orig_run_blocks = MM.MiniMaxH3Model.run_blocks
    if _orig_attn_forward is None:
        _orig_attn_forward = MM.Attention.forward
    overlap = os.environ.get("SP_OVERLAP") == "1"
    MM.Attention.forward = _sp_attn_forward_overlap if overlap else _sp_attn_forward
    MM.MiniMaxH3Model.run_blocks = _sp_run_blocks
    if RANK == 0:
        mode = f"overlap(P={_PIECES})" if overlap else "blocking"
        print(f"[sp_runtime] installed (WORLD={WORLD}) — run_blocks + Ulysses attention patched [{mode}]"
              + (" [SP_TIMING]" if _TIMING else ""), flush=True)
