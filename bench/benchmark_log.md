# MiniMax H3 denoiser — A100×4 (GPU 4,5,6,7) optimization benchmark log

Target: minimize ms per ONE H3 denoiser forward pass. NOT reducing steps.
Milestones: 1.5× / 2× / 2.5–3× vs baseline.

Env: torch 2.13.0+cu130, A100 80GB PCIe sm_80, bf16 ✓, SDPA ✓, Triton ✓, flash_attn/sage not standalone.
Base model in production: Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8 + ref2v_turbo_4step LoRA (4 steps).

Workload for baseline: 1280×720 × 124 frames (real fleet cut size) → packed seq S ≈ 42,000 tokens.

| Version | 1 denoise step (ms) | 4-step total | 8-step total | Peak VRAM | Speedup | Notes |
|---|---|---|---|---|---|---|
| baseline (1×A100, GPU7) | 18999 | ~76 s | ~152 s | ~30 GB | 1.00× | steady steps 1-3; step0=28s (alloc) |
| Ulysses SP 4×A100 (projected) | ~5700 | ~23 s | ~46 s | ~30 GB/GPU | ~3.3× | attention validated bit-exact, 2.93-3.14× |

## Per-component breakdown (baseline, one denoise forward, S≈42k, 50 blocks)
| component | ms | % of step |
|---|---|---|
| **attn.softmax (SDPA→FA-2)** | **13756** | **72.4%** |
| ffn.fc1 | 1540 | 8.1% |
| attn.qkv (fused) | 1349 | 7.1% |
| ffn.swiglu+down (fused) | 1076 | 5.7% |
| attn.out | 459 | 2.4% |
| attn.qknorm+rope (kitchen kernel) | 295 | 1.6% |
| adaln | 2.3 | 0.0% |
| other (norms/mod/embed) | 522 | 2.7% |

Key facts:
- SDPA already auto-selects **FlashAttention-2** (default≈FLASH in microbench) → single-GPU exact attention is ALREADY optimal; no free backend swap.
- Attention is O(S²)-compute-bound at S≈42k. The ~28% non-attention (GEMMs+adaln+norms) can't yield 1.5× while attention is 72%.
- **Only levers on the 72%: (A) 4-GPU sequence parallelism [Ulysses, EXACT, validated 2.93-3.14×], (B) SageAttention INT8 [approximate, quality-flagged, later].**

## Ulysses SP microbench (GPU4-7, heads=56 d=128 bf16), validated
- correctness: max_err=0.0 vs single-GPU FA-2 (bit-exact).
- S=40960: 253→86 ms/layer (2.93×); S=49152: 366→117 ms/layer (3.14×); all2all ~6 ms.

## Implementation results — 4-GPU Ulysses SP block loop (the 18.5s / 97% hot region)
Ground truth captured (plain base, no LoRA): h_in/h_out (40868, 5376), 13 packed segments.
| Run | 50-block loop ms | vs 1-GPU | correctness |
|---|---|---|---|
| Step B: 1-GPU replay (GPU4) | 18661 | 1.00× | **max_err 0.0 (bit-exact)** |
| Step C: 4-GPU SP (GPU7 contended by :8188) | 8840 | 2.11× | **max_err 0.0 (bit-exact)** |
| Step C: 4-GPU SP (clean, GPU4-7) | 8861 | **2.11×** | **max_err 0.0 (bit-exact)** |

Full denoise step projection: baseline 19.0s (18.66 block + ~0.34 prep/final) → SP ~8.86 block + ~0.34 prep + ~0.05 gather ≈ **~9.2s = ~2.06× per step, EXACT.** → hits "good result" (2×) milestone, bit-exact, on 4×A100.
Gap to ideal (~6.5-7s): straggler from contiguous-token sharding of the packed layout + 100 all-to-all sync bubbles + per-block launch overhead. Next levers: fuse q/k/v into one all-to-all, overlap comm, CUDA-graph the block loop.
Harness: /mnt/ssdraid/project/h3-opt/harness/sp_blockloop.py + run_sp.sh (one proc/GPU, NCCL Ulysses).
Sequence sharded by tokens (Sl=S/4), mod_segments+rope sharded to local range, Ulysses all-to-all around FA-2.

## Stretch investigation — topology is the wall (measured 2026-09-18)
GPU4-7 topology: **two NVLink pairs {4↔5}=NV12, {6↔7}=NV12; across pairs = PCIe (NODE, no NVLink).**
4-way SP component profile (block loop 8833ms): a2a_qkv 3383 (38%), softmax 3064 (35%), a2a_out 1170 (13%), ffn.fc1 347, qkv 266, ffn2 219, out 106, other 278. → **all-to-all = 4.5s = 51%, PCIe-bound.**
- Fused q/k/v all-to-all: 8921ms (no change → comm volume/PCIe, not call count, is the limit).
- 2-way SP on NVLink pair {4,5}: 9901ms=1.89×, comm only 0.54s, softmax-bound (6601ms).

Conclusion: **exact speedup is hardware-capped at ~2.1× on this 2-NVLink-pair box** (cross-pair PCIe floors 4-way all-to-all). Bit-exact 2.06× on the full step is the exact ceiling here.
Deployment note: 2-way-SP × 2 (one gen per NVLink pair) = same batch throughput as the current 4-way data-parallel fleet BUT ~1.9× lower latency per cut — best for oracle/interactive.
To exceed ~2.1× needs cutting comm VOLUME (approximate): FP8/INT8 all-to-all or SageAttention — quality-flagged, separate track.

### FP8 all-to-all experiment (approximate, 2026-09-18)
4-way SP, q/k/v transported in fp8_e4m3 (global per-tensor scale): **7370ms = 2.53× (full step ~2.47×)**.
comm a2a_qkv 3383→2175, a2a_out 1170→800. BUT **max_err=2.29e6 / mean 2.0e3 (ref absmax 16.9M) ≈ 14% — too lossy** (activation dynamic range too wide for one fp8 scale). Not acceptable as-is.
Refinements to try (each needs a full-generation visual check): per-head/per-token fp8 scale; or q,k in fp8 (RMSNorm'd, bounded) + v in bf16; or INT8 with finer scaling. Expected ~2.3-2.5× with acceptable error.

## CORRECTION — earlier 2.11× was NEIGHBOR-JOB CONTENTION, not the ceiling
The 8861ms (2.11×) runs were measured while the GPU0-3 job was at 100% util (PCIe/mem contention).
Clean measurement via the production seam (model.run_blocks patched by sp_runtime), GPU0-3 idle:
| run | run_blocks ms | speedup | correctness |
|---|---|---|---|
| seam SP 4-way #1 | 5793.7 | 3.22× | **max_err 0.0 (bit-exact)** |
| seam SP 4-way #2 | 5815.8 | 3.21× | bit-exact |
Full step ≈ 5.8s block + ~0.34 prep/final ≈ **~6.15s = ~3.1× per step, EXACT, bit-exact.**

## FINAL (this sprint) — STRETCH GOAL MET, EXACT
- **Exact 4-way Ulysses SP = ~3.2× block loop / ~3.1× full step, bit-exact (max_err 0.0).** Hits the 2.5-3× stretch with NO approximation. FP8 route abandoned (not needed).
- Real-world speedup ranges ~2.1× (box fully loaded by the other NUMA job) to ~3.2× (uncontended).
- Production foundation done: (1) behavior-safe `run_blocks` seam in model.py (verified: normal gen works), (2) reusable `sp_runtime.py` validated bit-exact through the seam.
- Remaining: persistent 4-rank SP service + oracle wiring; uneven-S sharding (currently requires S%WORLD==0).

## A-1 experiment — comm/compute overlap (exact, 2026-09-19)
Hypothesis: the 51%-comm figure (contended profile) means hiding the Q-input and O-output
all-to-alls under FlashAttention should recover a big chunk. Implemented in
`sp_runtime.py::_sp_attn_forward_overlap` (env `SP_OVERLAP=1`, `SP_PIECES=P`): K,V gathered
once (blocking), then Q-in and O-out all-to-alls split into P local-token pieces and software-
pipelined on a side CUDA stream so piece p+1's transfers overlap piece p's FA-2. Math unchanged.

Measured via the production seam (GPU4-7, GPU0-3 idle → **uncontended**), fleet paused + comfy VRAM freed:
| variant | run_blocks ms | vs 1-GPU | correctness |
|---|---|---|---|
| blocking (baseline) | 5666.7 | 3.29× | max_err 0.0 |
| overlap P=2 | 5681.5 | 3.28× | max_err 0.0 |
| overlap P=4 | 5725.7 | 3.26× | max_err 0.0 |

**Result: no speedup uncontended — in fact slightly slower, and monotonically worse with more
pieces (5666 < 5681 < 5725).** The overlap is correct (bit-exact) but the uncontended block loop
is **compute-bound**, not comm-bound: the 51% comm was specific to the *contended* run (GPU0-3
saturating PCIe). With GPU0-3 idle there is little comm to hide, and the extra kernel-launch /
stream-sync overhead of chunking dominates — hence more pieces = slower.

Implications:
- A-1 only pays off under **PCIe contention** (the real ~2.1× production regime). Could not measure
  that here without artificially loading GPU0-3 (another project's GPUs) — deferred.
- The "more pieces = slower" signal says **per-kernel launch overhead is a live cost** uncontended →
  **A-3 (CUDA-graph the 50-block loop)** is the better next lever for the uncontended case.
- Kept in the repo, gated behind `SP_OVERLAP=1` (default off; no change to the validated blocking path).

## Step caching — First-Block-Cache / TeaCache (approximate, 2026-09-19)
Orthogonal to SP: skip recompute across denoise STEPS instead of parallelizing one step.
Unlike SP it consumes no extra GPUs → multiplies the data-parallel fleet's throughput.
`custom_nodes/h3_fbcache/` (env `H3_FBCACHE=1`): run only block 0 each step, measure its
residual's rel-L1 vs last step; while accumulated < threshold, skip blocks[1:] and reuse the
cached rest-of-stack residual. Step 0 always computes. New-gen detected via timestep rising.

Real A/B, same backend (GPU5/:8195), same seed 740049, full-res 1280×720 × 158f, 6-step turbo:
| run | wall-clock | skips | note |
|---|---|---|---|
| cache OFF | **212 s** | 0/6 | baseline |
| cache ON (thresh 0.08) | **92 s** | **5/6** | 2.3× — but see below |

**2.3× speedup is real, but the naive signal OVER-SKIPS.** The per-step log showed
`step0 calc; steps1-5 SKIP; accum=0.000` — i.e. the rel-L1 signal read ~0 every step and
skipped all but step 0, collapsing quality toward a 1-step generation. Root cause: the signal
is a **mean over the whole packed sequence** `[text|cond|audio|video]`; the static conditioning
tokens dominate the mean, hiding the video tokens' real change. (Takes: cut2162 kf4 = ON,
kf5 = OFF, for visual comparison.)

Fix (TODO before any deploy):
- Restrict the change signal to the **video/audio (denoised) tokens only** (needs the layout's
  video/audio ranges plumbed into run_blocks), or
- Use a **t_emb-based TeaCache signal** (t_emb changes every step; add a per-model polynomial
  rescale), or
- Simply lower the threshold — but the mean signal is so flat (0.000) that thresholding alone
  won't give a sensible skip pattern; the signal itself must change.
Expected with a correct signal on the 6-step turbo path: ~1.3–1.6× at good quality (few steps
to skip); much larger on long-step paths (30-step audio/no-turbo). Node is gated (default off).

Note: the audio-regen path (0.5× res, 30-step) showed NO wall-clock gain from caching (62 s on
vs 62 s off, same backend) — that path is model-load/VAE/mux-bound, not denoise-bound.

## Strategy (revised by measurement)
Attention >50% → prioritize attention. Single-GPU exact maxed → implement **4-GPU Ulysses sequence-parallel denoiser** (exact). Then AdaLN precompute + rope cache + CUDA graph for the residual overhead; SageAttention as a separate quality-flagged experiment.

## Correctness log (max/mean error vs baseline)
(pending)
