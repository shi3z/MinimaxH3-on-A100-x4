# MiniMax‑H3 on A100 ×4 — exact denoiser acceleration

Making **one denoiser forward pass** of the MiniMax‑H3 DiT as fast as possible on
**4× A100 80GB (sm_80)**, *without* reducing denoising steps and *without* any
approximation to the attention. Result: **~3.1× per step, bit‑exact (max_err 0.0)**
via 4‑way Ulysses sequence parallelism.

> Philosophy transferred from [shi3z/deepseekv4.1-A100-custom](https://github.com/shi3z/deepseekv4.1-A100-custom):
> profile first, find the real bottleneck, attack it exactly, verify correctness
> (max/mean error) after **every** change, and benchmark the whole step — not a
> microkernel in isolation.

## The rules we held ourselves to

1. **Don't cut steps.** Acceleration LoRAs already reduce step *count*; this work
   makes each remaining step cheaper.
2. **Exact before approximate.** No sparse/linear/INT8 attention until the exact
   path is fully optimized. (We tried FP8 all‑to‑all — abandoned, see below.)
3. **Verify every change.** Every variant is checked `max_err`/`mean_err` against
   the single‑GPU FlashAttention‑2 baseline. "Fast but wrong" is a failure.
4. **Benchmark the whole denoise step**, at the real fleet workload
   (1280×720 × 124 frames → packed sequence S ≈ 42k tokens).
5. **GPU 4,5,6,7 only.**

## The model (what we're accelerating)

MiniMax‑H3 DiT: 50 blocks, hidden 5376, 56 heads × head_dim 128 (inner 7168),
SwiGLU FFN (14336), fused QKV, per‑head qk‑RMSNorm+RoPE fused kernel, AdaLN,
packed sequence `[text | cond | audio | video]`, INT8 (w4a4) quant.
Attention = ComfyUI `optimized_attention` → SDPA (auto‑selects FlashAttention‑2 on A100).

## Where the time goes (baseline, 1× A100, one forward, S≈42k, 50 blocks)

| component | ms | % of step |
|---|---|---|
| **attn.softmax (SDPA→FA‑2)** | **13756** | **72.4%** |
| ffn.fc1 | 1540 | 8.1% |
| attn.qkv (fused) | 1349 | 7.1% |
| ffn.swiglu+down (fused) | 1076 | 5.7% |
| attn.out | 459 | 2.4% |
| attn.qknorm+rope (kitchen kernel) | 295 | 1.6% |
| adaln | 2.3 | 0.0% |
| other (norms/mod/embed) | 522 | 2.7% |

**Attention is 72%** and it's O(S²)‑compute‑bound at S≈42k. SDPA already picks
FlashAttention‑2, so single‑GPU exact attention is *already optimal* — there is no
free backend swap, and the 28% of non‑attention GEMMs can't buy 1.5× on their own.
The only exact lever on the 72% is **spreading the sequence across GPUs**.

## The approach — 4‑way Ulysses sequence parallelism (exact)

Shard the packed sequence by tokens (`Sl = S/W`), all‑to‑all seq→head so each GPU
holds `Hl = H/W` heads over the *full* sequence, run FlashAttention‑2, all‑to‑all
head→seq. This **redistributes the identical compute** — it is bit‑exact, not an
approximation. Implemented as a monkeypatch over a behavior‑preserving `run_blocks`
seam added to the model, so production code is untouched.

- `harness/sp_runtime.py` — the reusable runtime: patches `MiniMaxH3Model.run_blocks`
  (shard → 50 blocks with Ulysses attn → all‑gather) and `Attention.forward`
  (all‑to‑all around FA‑2). One process per GPU, NCCL.
- `patches/model_run_blocks_seam.md` — the exact, behavior‑identical seam to add to
  `comfy/ldm/minimax/model.py` so the runtime has a clean override point.
- `custom_nodes/h3_profiler/` — env‑gated profiler (`H3_PROFILE=1`) + I/O ground‑truth
  capture (`H3_DUMP=1`) for the correctness harness.

## Results

| Version | 1 denoise step | Speedup | Correctness |
|---|---|---|---|
| baseline (1× A100) | 18999 ms | 1.00× | — |
| **4‑way Ulysses SP (uncontended)** | **~6.15 s** | **~3.1×** | **max_err 0.0 (bit‑exact)** |
| 4‑way SP (box loaded by other‑NUMA job) | ~9.2 s | ~2.1× | bit‑exact |
| 2‑way SP on one NVLink pair | ~9.9 s | ~1.9× | bit‑exact |

Validated through the production `run_blocks` seam, GPU0‑3 idle:
`run_blocks 5793.7 / 5815.8 ms` vs `18661 ms` single‑GPU → **3.21–3.22×, max_err 0.0**.

### Topology is the wall

GPU4‑7 = two NVLink pairs `{4↔5}=NV12`, `{6↔7}=NV12`; across pairs = PCIe (no NVLink).
Under contention the 4‑way all‑to‑all is PCIe‑bound (~51% of the step). So the honest
number is a **range: ~2.1× when the box is loaded, ~3.2× uncontended**, all bit‑exact.
Deployment note: 2‑way‑SP × 2 (one gen per NVLink pair) matches the data‑parallel
fleet's throughput but gives ~1.9× lower single‑cut latency — best for interactive use.

### Dead ends (recorded on purpose)

- **Fused q/k/v all‑to‑all**: no change (8921 vs 8861 ms) — the limit is comm
  *volume* over PCIe, not all‑to‑all call count.
- **FP8 all‑to‑all** (approximate): 2.53×, but `max_err 2.29e6 / ref absmax 16.9M ≈ 14%`
  — a single global fp8 scale underflows the activation dynamic range. Abandoned;
  the exact 3.1× made it unnecessary.

Full numbers and method: [`bench/benchmark_log.md`](bench/benchmark_log.md).

## Layout

```
harness/    sp_runtime.py (reusable), sp_blockloop.py, sp_seam_test.py, run_*.sh
bench/      benchmark_log.md, ulysses_attn_microbench.py, capture_gen.py
patches/    model_run_blocks_seam.md (the behavior-identical seam)
custom_nodes/h3_profiler/  env-gated profiler + I/O capture
logs/       measurement logs (the 855MB *.pt I/O ground truth is gitignored)
```

## Reproduce

```bash
# 1. add the run_blocks seam (patches/model_run_blocks_seam.md) to model.py
# 2. capture I/O ground truth once
H3_DUMP=1 python bench/capture_gen.py           # -> logs/io_blockloop_{in,out}.pt
# 3. verify bit-exact + benchmark through the seam, 4 GPUs
bash harness/run_seam.sh                          # expects max_err 0.0
# microbench attention alone:
bash harness/run_sp.sh                            # 4-way GPU4-7
bash harness/run_sp2.sh                           # 2-way NVLink pair
```

Env: torch 2.13.0+cu130, A100 80GB PCIe sm_80, bf16, SDPA/FA‑2, Triton.

## Layer-selective SLA presets (approximate, opt-in)

Unlike everything above, these trade exactness for speed. They are **off by default**. Nothing changes unless a preset is requested.

| preset | SLA25 layers (rest SLA15) | research name | gen time* | vs Dense |
|---|---|---|---|---|
| `quality` | 31-42, 46-49 | E8 | ~27.8 s | ~1.60x |
| `balanced` | 31-42, 48-49 | T2 | ~27.4 s | ~1.62x |
| `speed` | 34-42 | E3 | ~26.7 s | ~1.67x |

\*Measured on A100 SP-2 (GPU 4+5), ref2va turbo 4-step, 1280x720x158. Dense ~44.5 s, SLA25 on all layers ~33.0 s.

- **quality**: approximately comparable to SLA25 on all layers in our validation set, at ~1.6x Dense speed.
- **balanced**: nearly the quality preset. Slightly more fine-texture / night-scene noise risk.
- **speed**: the fastest recommended preset. It has a visible quality trade-off in large motion, scene structure and fine detail. It is not equal quality.

The validation set is finite: 4 clips (face, large motion, night crowd/fine texture, anime) × 3 seeds. There is no claim of universal superiority over SLA25 on all layers.

**Where the presets live.** The schedules are defined once, in `comfy-h3/custom_nodes/h3_sla` (`PRESETS`). A mirror of that node is kept in this repo at `custom_nodes/h3_sla/`; deploy it by copying it into `comfy-h3/custom_nodes/`. How to select one:

- **ComfyUI node:** `H3SLAPatch(preset=quality|balanced|speed|custom, layer_keep="25:31-42,46-49")`. With `preset=off`, `topk` is uniform, which is the legacy behaviour.
- **Fleet job JSON:** `"sla_preset": "quality"`. `custom` also needs `"sla_layer_keep": "..."`.
- **CLI:** `harness/sp_generate.py <job.json> <tag> --sla-preset quality` or `--sla-layer-keep "25:31-42,46-49"`.

**SP runtime.** `harness/sp_runtime.py` applies the same schedule inside Ulysses attention. It does so only when a preset is active, and uses the blocking path. Without a preset, SP attention stays FA2, as before.

**Logging and dashboard.**
- At each generation start, rank 0 prints the preset, the default keep, the SLA25 ranges and the predicted cost.
- Status goes to `sp_status/h3_sla/*.json`. The dashboard panel "SLA preset" reads it (`/api/sla_preset`).
- With `H3_SLA_TIMING=1` the panel also shows per-layer attention ms.

**Predictors.** None. The per-token residual predictors, Global Memory and Engram are research-only and are never loaded.

**Research record.** See `/mnt/ssd/project/newh3/experiments/h3_token_residual/PRESETS.md`. The regression test reproduces the research latents bit-exactly.
