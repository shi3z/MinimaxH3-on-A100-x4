# H3 Optimization Dashboard (lightweight, tailnet, iPad)

Live view of MiniMax-H3 optimization on tsuginosuke. No Grafana/Prometheus — stdlib only.

## Run
    python3 dashboard/server.py --port 8765 --bind auto
Binds to the Tailscale IP (falls back to 0.0.0.0). Open on the iPad (same tailnet):
    http://<tailscale-ip>:8765     e.g. http://100.126.237.55:8765
Landscape. Live via SSE.

## Panels
- Live generation overview (gen id, elapsed, step/total, layer, seq_len, kernel, dtype, ETA, step time)
- Per-stage timing (horizontal breakdown, ms + %)
- Attention detail (QK/softmax/PV/cp.async wait/barrier — when the custom kernel emits them)
- GPU monitor (all A100: util/mem/temp/power/clock via nvidia-smi @2s; SM/TC util = explicit Nsight mode)
- Kernel benchmark history (never overwrites the historical best; highlights current/fastest/vs-FA2)
- Timeline (last 100 measurements; spikes/regressions at a glance)

## Instrument inference (see dashboard/h3_metrics.py)
    export H3_METRICS=1 H3_DASH_URL=http://100.126.237.55:8765
    import h3_metrics as M
    M.set_context(gen_id=..., kernel="FA3_SM80_H3_v1", seq_len=N, step=s, total_steps=50, layer=l)
    with M.stage("attention"): ...        # CUDA-event timed, async-collected (no hot-path sync)
    M.emit("total_step", ms)              # feeds the timeline
    M.record_bench("FA3_SM80_H3_v1", latency_ms, vs_fa2=..., config="...")
Correctness: GPU timing uses torch.cuda.Event; normal mode never syncs per op. H3_METRICS_DEBUG=1
allows hard-synced sub-timings for a deeper profiling pass. Disabled (H3_METRICS unset) = ~zero overhead.
