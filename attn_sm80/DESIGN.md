# FA3_SM80_H3 — FlashAttention-3-style scheduling ported to A100 / SM80, specialized for MiniMax-H3

Goal: beat the FA-2 BF16 baseline (**263 ms/layer**, cuDNN 281 ms) for the H3 attention at its
**fixed geometry**: B=1, H=56, head_dim **D=128**, N≈**40,868** (long-context prefill), BF16,
non-causal, no mask. NOT a generic kernel — every choice is hard-wired to this shape.

## Roofline (why there is headroom at all)
FLOPs = 4·N²·D·H = 4.79e13. FA-2 does it in 263 ms → **182 TFLOPS = 58% of the 312 TFLOPS bf16
peak**. So ~42% is lost to softmax (exp on CUDA cores), non-MMA cycles, and sync. FA-3's Hopper
gains close much of that via async WGMMA + TMA. On SM80 those exact mechanisms are unavailable;
the question is how much of the *scheduling idea* survives.

## FA-3 pipeline decomposed → portable vs Hopper-only
| FA-3 mechanism | Purpose | SM80 status | SM80 replacement |
|---|---|---|---|
| **WGMMA** (async warpgroup MMA) | issue MMA, continue, overlap | **Hopper-only** | `mma.sync.aligned.m16n8k16` (synchronous, but hardware-pipelined) |
| **TMA** (tensor mem accel copy) | async bulk global→shared | **Hopper-only** | `cp.async.cg` (Ampere async copy) |
| **setmaxnreg** (dyn regs per WG) | give consumer more regs | **Hopper-only** | fixed reg budget; tune occupancy |
| **Warpgroup producer/consumer specialization** | one WG loads, others compute | **portable idea** | warp specialization: producer warps issue cp.async, consumer warps mma.sync, sync via `bar.sync`/named barriers + smem mbarrier-style flags |
| **Ping-pong 2-stage GEMM↔softmax overlap** | hide exp behind MMA | **portable idea** | two consumer warp-groups alternate: WG-A does PV(tile i) while WG-B does softmax(tile i+1) |
| **Intra-loop QK/softmax/PV software pipeline** | overlap within a warp | **portable, already in FA-2** | multi-stage cp.async + prefetch next K/V tile during current MMA |
| Register-resident Q | avoid reloading Q | portable | keep Q tile in regs across the whole KV loop |
| FP8 + incoherent processing | 2× throughput | Hopper fp8 | N/A (bf16 target) |

**Key insight from measurement**: the *intra-loop software pipeline* (cp.async multistage +
register-resident Q) is exactly what a tuned Triton kernel expresses — and it lands at **0.76×**
(345 ms), i.e. FA-2 already does this better. Therefore the ONLY differentiator left on SM80 is the
one Triton cannot express: **explicit warp specialization + ping-pong softmax/GEMM overlap**. That
is the whole point of FA3_SM80_H3.

## Answers to the 12 investigation points (for this geometry)
1. **Producer/consumer warp specialization** — yes, the core bet. 1 producer warp-group (4 warps)
   drives cp.async K/V staging; 2 consumer warp-groups do mma.sync QK/PV. This is what might beat FA-2.
2. **cp.async staging of K/V** — `cp.async.cg.shared.global` 16B/thread, per KV tile (BLOCK_N × 128).
3. **Double/triple buffering** — A100 smem 164 KB usable/CTA. K+V tile bf16 (BLOCK_N=64): 64·128·2·2 =
   32 KB/stage → triple-buffer (96 KB) fits with Q (128·128·2 = 32 KB regs/smem). Start double, try triple.
4. **QK matmul ↔ softmax overlap** — consumer WG-B computes exp/rowmax/rowsum of S(i) while WG-A
   issues QK mma for S(i+1). Ampere overlap is scheduler-level (warp interleave), not WGMMA-async.
5. **softmax ↔ PV matmul overlap** — the online-softmax rescale (acc *= alpha) runs on CUDA cores
   while the other WG issues PV mma; ping-pong.
6. **Reduced synchronization** — named barriers (`bar.sync 0..15`) between producer/consumer instead of
   full `__syncthreads()`; only sync on buffer ready/consumed.
7. **Register-resident Q** — Q tile (BLOCK_M×128) loaded once via ldmatrix into fragments, reused for
   all KV tiles. No reload.
8. **Minimized smem round trips** — S (scores) kept in registers/mma accumulators, not written to smem
   between QK and PV where possible; only K/V go through smem (from cp.async).
9. **Persistent CTA** — with N≈40k and 56 heads, grid = ceil(N/BLOCK_M)·56 ≈ 320·56 ≈ 17,920 CTAs for
   BLOCK_M=128 → far exceeds SM count (108); a persistent-CTA (grid = #SMs, loop over tiles) can cut
   launch/scheduling overhead and keep smem buffers warm. Try after correctness.
10. **Block scheduling for N≈40k** — BLOCK_M=128 (Q rows), BLOCK_N=64 (KV). 320 M-tiles × 56 heads.
    Order heads innermost per SM for L2 reuse of nothing (Q differs per tile) — schedule by (head, m-tile).
11. **Split-K / seq partitioning across 56 heads** — non-causal, N huge, so each (m-tile,head) is
    independent and already plenty of parallelism (17,920 tiles ≫ 108 SMs). Split-K over the KV
    dimension is NOT needed (enough CTAs); it would only add reduction cost. Skip.
12. **Occupancy vs registers** — D=128 accumulators (BLOCK_M×128 fp32 = big reg pressure). Target ≥2
    CTAs/SM. Q frags + acc frags dominate regs; balance BLOCK_M (64 vs 128) against occupancy.

## Implementation plan (staged, correctness-first)
- **v0**: single-warp-group flash with `cp.async` double-buffered K/V + `mma.sync m16n8k16` QK & PV,
  online softmax, register-resident Q, D=128/bf16/non-causal hard-wired. Verify rel-err < 5e-3 vs
  SDPA-FLASH, benchmark. (Establishes: does raw CUDA reach FA-2 at all?)
- **v1**: add producer/consumer **warp specialization** (cp.async producer WG + mma consumer WGs).
- **v2**: add **ping-pong** softmax↔PV overlap across two consumer WGs; named-barrier sync.
- **v3**: persistent CTA + triple buffering + occupancy tuning.
Benchmark every stage vs FA-2 263 ms. Success = beat 263 ms wall-clock at rel-err < 5e-3.

## Build/runtime
torch 2.13.0+cu130, `CUDA_HOME=/usr/local/cuda-12.9`, `-arch=sm_80`, `torch.utils.cpp_extension`.
Verified: a trivial sm_80 extension builds + runs (max_err 0.0). CUDA path is viable.
Deliverable: a switchable backend `attn(q,k,v, backend=...)` selecting {sdpa_flash | fa3_sm80_h3}.
