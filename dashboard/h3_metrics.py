"""Lightweight in-process metrics collector for the H3 optimization dashboard.

Design goals (per spec):
- Emit small events: {ts, gen_id, step, layer, stage, kernel, duration_ms, seq_len, dtype, gpu_id}.
- GPU timing via torch.cuda.Event (NOT python wall-clock around async CUDA work).
- Collect completed CUDA-event timings asynchronously (poll Event.query()), so normal mode does
  NOT force torch.cuda.synchronize() after every op. A DEBUG_PROFILING mode allows hard syncs.
- Trivial enable/disable; near-zero overhead when disabled.
- Events are POSTed (batched, background thread) to the dashboard server /emit; if the server is
  down, events are dropped silently (never blocks/breaks inference).

Enable with env H3_METRICS=1. Point at the dashboard with H3_DASH_URL (default http://127.0.0.1:8765).
DEBUG_PROFILING via H3_METRICS_DEBUG=1.
"""
import os, time, json, threading, collections, urllib.request

ENABLED = os.environ.get("H3_METRICS", "0") == "1"
DEBUG = os.environ.get("H3_METRICS_DEBUG", "0") == "1"
DASH_URL = os.environ.get("H3_DASH_URL", "http://100.126.237.55:8770").rstrip("/")  # tailnet dashboard (8765 reserved)
GPU_ID = int(os.environ.get("H3_METRICS_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"))
FILE_SINK = os.environ.get("H3_METRICS_FILE")  # optional JSONL capture of every event

_ctx = {"gen_id": None, "step": 0, "total_steps": 0, "layer": 0, "seq_len": 0, "kernel": "FA2", "dtype": "bf16"}
_send_q = collections.deque(maxlen=20000)
_pending = collections.deque()          # (start_evt, end_evt, meta) awaiting completion
_lock = threading.Lock()
_evt_pool = []

def set_context(**kw):
    _ctx.update({k: v for k, v in kw.items() if v is not None})

def _post(path, obj):
    try:
        req = urllib.request.Request(DASH_URL + path, data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=0.5).read()
    except Exception:
        pass  # dashboard optional; never break inference

def _event(stage, duration_ms, ctx, extra):
    return {"ts": time.time(), "gen_id": ctx["gen_id"], "step": ctx["step"],
            "total_steps": ctx["total_steps"], "layer": ctx["layer"], "stage": stage,
            "kernel": ctx["kernel"], "duration_ms": round(float(duration_ms), 4),
            "seq_len": ctx["seq_len"], "dtype": ctx["dtype"], "gpu_id": GPU_ID, **extra}

def emit(stage, duration_ms, **extra):
    """Emit a completed measurement event using the CURRENT context (for synchronous callers)."""
    if not ENABLED:
        return
    _send_q.append(_event(stage, duration_ms, dict(_ctx), extra))

# ---- CUDA-event based timing (async collection, no per-op sync) ----
def _mk_events():
    import torch
    if _evt_pool:
        return _evt_pool.pop()
    return (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))

class stage:
    """`with stage("attention"):` — brackets a GPU region with CUDA events; the elapsed time is
    collected later (asynchronously) so we never sync on the hot path (unless DEBUG_PROFILING).
    The context (gen_id/step/layer/...) is snapshotted at ENTER, so the async-collected event is
    labeled with the state at measurement time, not at collection time."""
    def __init__(self, name, **extra):
        self.name = name; self.extra = extra; self.s = self.e = None; self.ctx = None
    def __enter__(self):
        if not ENABLED:
            return self
        self.ctx = dict(_ctx)
        self.s, self.e = _mk_events(); self.s.record(); return self
    def __exit__(self, *a):
        if not ENABLED or self.s is None:
            return False
        import torch
        self.e.record()
        if DEBUG:
            torch.cuda.synchronize()
            _send_q.append(_event(self.name, self.s.elapsed_time(self.e), self.ctx, self.extra))
            _evt_pool.append((self.s, self.e))
        else:
            with _lock:
                _pending.append((self.s, self.e, self.ctx, self.extra, self.name))
        return False

def _collector():
    import torch
    while True:
        # 1) drain finished CUDA events without global sync
        ready = []
        with _lock:
            n = len(_pending)
        for _ in range(n):
            with _lock:
                if not _pending:
                    break
                s, e, ctx, extra, name = _pending.popleft()
            try:
                if e.query():                       # completed -> safe to read, no sync
                    _send_q.append(_event(name, s.elapsed_time(e), ctx, extra))
                    _evt_pool.append((s, e))
                else:
                    with _lock:
                        _pending.append((s, e, ctx, extra, name))
            except Exception:
                pass
        # 2) flush emitted events to the dashboard in a batch
        batch = []
        while _send_q and len(batch) < 500:
            batch.append(_send_q.popleft())
        if batch:
            _post("/emit", {"events": batch})
            if FILE_SINK:
                try:
                    with open(FILE_SINK, "a") as f:
                        for ev in batch:
                            f.write(json.dumps(ev) + "\n")
                except Exception:
                    pass
        time.sleep(0.25)

def record_bench(kernel, latency_ms, vs_fa2=None, config="", note=""):
    """Post a kernel benchmark result to the dashboard history (never overwrites the best)."""
    _post("/bench", {"kernel": kernel, "latency_ms": round(float(latency_ms), 3),
                     "vs_fa2": vs_fa2, "config": config, "note": note, "ts": time.time()})

if ENABLED:
    threading.Thread(target=_collector, daemon=True).start()
