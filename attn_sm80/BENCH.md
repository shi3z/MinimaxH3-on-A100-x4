# FA3_SM80_H3 benchmark log

Geometry: B=1 H=56 D=128 N=40868 BF16 non-causal. GPU: A100 80GB PCIe (GPU6), sm_80.
Env: torch 2.13.0+cu130, CUDA 12.9 nvcc, -arch=sm_80.
Baseline (live, measured): **FA-2 SDPA-FLASH = 263.1 ms/layer** (cuDNN ~281 ms).
Correctness tol: rel_mean < 5e-3 vs SDPA-FLASH.

## Results

| version | latency (ms) | vs FA2 | rel_err | regs | smem/CTA | CTAs/SM | notes |
|---|---|---|---|---|---|---|---|
| FA2 (SDPA FLASH) | 263.1 | 1.00x | (ref) | - | - | - | reference / target |
| v0 cp.async single-buf | 1486.0 | 0.18x | 2.28e-3 | 179 | 56 KB | 2 | BM=BN=64, 4 warps, no load/compute overlap |

## Diagnoses & fixes

### v0 (single-buffered)  — 1486 ms, 0.18x  (CORRECT)
Fragment plumbing validated by micro-tests (QK rel 1e-7, PV rel 5e-8).
Dominant cost: **no load/compute overlap** — cp.async immediately followed by wait_group 0
every iteration (639 iters), so global-load latency is fully exposed. Compounded by
**low occupancy** (2 CTAs/SM = 8 warps/SM = 12.5%) which cannot hide that latency.
Fix path: double-buffer K/V (prefetch tile j+1 while computing tile j); reuse sQ region
for sP to hold smem at 80 KB (keeps 2 CTAs/SM).
