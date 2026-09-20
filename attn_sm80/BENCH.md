# FA3_SM80_H3 benchmark log

Geometry: B=1 H=56 D=128 BF16 non-causal. GPU: A100 80GB PCIe (GPU6), sm_80.
Env: torch 2.13.0+cu130, CUDA 12.9 nvcc, -arch=sm_80.
Correctness tol: rel_mean < 5e-3 vs SDPA-FLASH.  Profiling: ncu blocked (ERR_NVGPUCTRPERM)
-> used in-kernel clock64() phase timing (PROFILE build) + cudaFuncGetAttributes +
cudaOccupancyMaxActiveBlocksPerMultiprocessor for occupancy. CLEAN build (no clock64) for all
latency numbers.

Baselines (measured live on this box):
- N=40868: FA-2 SDPA-FLASH = 263 ms/layer.
- N=46052: FA-2 anchor = 342.1 ms/layer (coordinator), 50-layer step = 22891 ms.

## Headline
CORRECT v0 kernel (rel 2.28e-3). Best config **BM=64 BN=64 STAGES=1 (single-buf) PAD=8 MINCTA=3**:
- N=40868: **359 ms = 0.73x** of FA-2 (263 ms).
- N=46052: see table row (CLEAN).
Did NOT beat FA-2. Closest = 0.73x. Honest profiler reason below.

## Optimization journey (N=40868, CLEAN latency)

| step | change | ms | vs FA2 | rel_err | note |
|---|---|---|---|---|---|
| FA-2 | SDPA FLASH reference | 263 | 1.00x | - | target |
| v0.0 | single-buf cp.async + mma, P via smem | 1486 | 0.18x | 2.28e-3 | ldmatrix bank conflicts dominate |
| v0.1 | + double-buffer K/V | 1472 | 0.18x | 2.28e-3 | no gain: not load-bound yet |
| v0.2 | + register-resident P (no smem roundtrip) | 1348 | 0.20x | 2.28e-3 | S-acc layout == PV A-operand layout |
| v0.3 | + batched ldmatrix (decouple from mma) | 1358 | 0.19x | 2.28e-3 | small |
| **v0.4** | **+ PAD=8 smem (kill ldmatrix bank conflicts)** | **363** | **0.72x** | 2.28e-3 | **4x jump — conflicts were THE bottleneck** |
| v0.5 | single-buf + MINCTA=3 (3 CTA/SM) tuning | 359 | 0.73x | 2.28e-3 | best |

## Diagnosis-driven tuning (each measured, N=40868)
- double-buffer (STAGES=2, 68KB) = 405 ms (0.65x): overlap gain < occupancy loss (2 vs 3 CTA). REJECTED.
- BN=32 double-buffer (34KB, 3 CTA) = 410 ms: smaller tiles raise compute floor > overlap saved. REJECTED.
- BM=128 (halve L2 re-reads) = 395-432 ms: occupancy drop (1-2 CTA) > L2 saving. REJECTED.
- PAD sweep: PAD=8 (363) == PAD=24 (364) best; PAD=16 (435) hits a 2-way conflict; PAD=4 crashes
  (misaligned cp.async — SD must stay 16B aligned -> PAD multiple of 8).
- MINCTA: 3 best (359); 2 -> 421 (fewer CTAs); 4 -> 509 (register spill).

## Profiling table (required)  [N=40868, CLEAN latency; sub-stage % from PROFILE build clock64]

| version | latency | vs FA2 | TC util* | occupancy | regs | smem/CTA | dominant stall |
|---|---|---|---|---|---|---|---|
| FA2 (SDPA) | 263 ms | 1.00x | ~58% | (n/a) | - | - | reference |
| v0 cp.async (no pad) | 1486 ms | 0.18x | ~11% | 12.5% (2 CTA) | 179 | 56 KB | ldmatrix smem bank conflicts (8-way) |
| **v0 (PAD=8, best)** | **359 ms** | **0.73x** | ~43% | 18.8% (3 CTA) | 168 | 34 KB | exposed cp.async/L2 load wait (38%) |
| v1 warp-specialized | not built | - | - | - | - | - | (see status) |
| v2 ping-pong | not built | - | - | - | - | - | (see status) |

*TC util = mma-throughput-floor(154 ms) / measured, rough.

Per-CTA sub-stage split (PROFILE build, best config): cp_async_wait 38%, qk_gemm 26%,
softmax 18%, pv_gemm 19%.

## Root-cause of the remaining gap (measured, not inferred)
NOLOAD experiment (skip cp.async+wait, compute on resident smem): CLEAN = **260 ms** vs full 364 ms.
=> compute floor alone (260) ~ FA-2 (263); exposed load adds ~104 ms.
- The load is L2-bound, not HBM: one head's K+V = 21 MB fits A100 40 MB L2, so the 639 m-tiles/head
  re-read K,V from L2 (~1.1 TB of L2 traffic).
- To beat FA-2 both must improve: (a) hide the 104 ms load AND (b) cut the 260 ms compute floor.
  Every buffering scheme that hides load (STAGES=2, BN=32, BM=128) drops occupancy from 3->1-2 CTA,
  which RAISES the compute floor more than the load it hides. So single-buffer/3-CTA is the optimum
  for this single-warp-group design.
- The compute floor's excess over the mma floor is ~72 ms of softmax that runs serially between
  QK and PV mma. Overlapping it (v2 ping-pong: one warp-group does softmax while another does mma)
  is the one lever left, and the one a generic Triton kernel cannot express (design premise).

## Apples-to-apples table @ S=46052 (coordinator's anchor), CLEAN build

| version | attn ms | step ms (50L) | vs FA2 | correctness | regs | smem/CTA | occupancy |
|---|---|---|---|---|---|---|---|
| FA2 (SDPA-FLASH) | 342.1 (anchor; 334.4 measured here) | 22891 | 1.00x | 0 | n/a | n/a | n/a |
| FA3_SM80_H3 v0 | 457.9 | 28682 (proj) | 0.747x | 2.29e-3 PASS | 168 | 34 KB | 0.188 (3 CTA/SM) |

Per coordinator's reject rule (>10% over FA2 = >376 ms): v0 at 457.9 ms does NOT clear the bar.
Reported honestly as the current best; it is correct, no spill, 3-CTA resident.

## v2 ping-pong / software-pipelined (BUILT + MEASURED)
Kernel `fa3_pp_kernel`: double-buffered K/V, prefetch of tile i+1 issued before softmax(i)+PV(i)
so the cp.async load overlaps compute; register-resident P; PAD=8; named-per-tile barriers.

| version (config) | attn ms N=40868 | vs FA2 | attn ms S=46052 | vs FA2 | rel_err | regs | smem | occ |
|---|---|---|---|---|---|---|---|---|
| v0 (BN=64 STAGES=1 3CTA) | 358 | 0.74x | 458 | 0.75x | 2.28e-3 | 168 | 34KB | 0.188 |
| v2 BN=64 STAGES=2 (2 CTA) | 401 | 0.66x | - | - | 2.28e-3 | ~162 | 68KB | 0.125 |
| v2 BN=32 STAGES=2 (3 CTA) | **395** | **0.666x** | **507** | **0.674x** | 2.28e-3 | 162 | 34KB | 0.188 |

**v2 is a REJECT** (395 ms > FA-2 265 ms; also worse than v0's 358). BUT the profile proves the
mechanism worked as designed:

PROFILE substage split, v2 BN=32 (clock64): cp_async_wait **3.8%** (v0 was 38% — LOAD IS NOW HIDDEN),
qk_gemm 45%, softmax 30%, pv_gemm 21%.

Root cause of the v2 loss (measured, honest):
- Hiding the load requires 2 K/V buffers. At BN=64 that is 68 KB -> only 2 CTA/SM (occ 0.125) -> 401 ms.
- Forcing 3 CTA back requires BN=32 (34 KB), but BN=32 halves tile size and doubles iteration/barrier
  count, which inflates the *compute* floor: qk_gemm 26%->45%, softmax 18%->30% of a now-slower loop.
- Net: the ~35% load we hid is more than eaten by the compute-floor increase. Same wall as v0 tuning.
- Also confirms softmax was already largely hidden at v0's 3-CTA/12-warp occupancy (the HW scheduler
  interleaves 12 independent warps), so an explicit softmax/mma ping-pong has little headroom on SM80.

## v1 status
Not built as a separate kernel: the v2 result shows the binding constraint is register-driven
occupancy (the 64-reg O accumulator), which warp-specialization (dedicating warps to cp.async) would
worsen by removing compute warps. The profile does not support it as a win path on this geometry.

## CONCLUSION (honest)
Best measured: **v0 = 358 ms (N=40868, 0.74x) / 458 ms (S=46052, 0.747x), correct.** Did NOT beat FA-2.
The single-warp-group design's ceiling is set by the 64-register O accumulator -> max ~3 CTA/SM.
Every load-hiding / softmax-overlap scheme (double buffer, BN=32 pipeline, BM=128) either drops
occupancy or shrinks tiles, raising the compute floor more than it saves. v2 pipelining *did* hide
the load (38%->3.8%) but could not convert it to a net win. Beating FA-2 (which itself sits at the
~260 ms compute floor here) would require breaking the register/occupancy wall — a different
accumulator strategy or Hopper-class async WGMMA that SM80 lacks.

## Files
- fa3_sm80_h3.cu  — kernel (mma.sync.m16n8k16, ldmatrix, cp.async.cg; PROFILE/PAD/STAGES/MINCTA/BM/BN compile knobs)
- attn_sm80.py    — backend switch attn(q,k,v, backend="fa3_sm80_h3"|"sdpa_flash"); default = best config
- bench_attn.py   — harness (given)
- microtest.py    — QK/PV tile-GEMM fragment validation (rel ~1e-7)
- bench_variants.py, report_metrics.py — sweep + dashboard posting
