#!/usr/bin/env python3
"""H3 optimization live dashboard — lightweight, dependency-free (stdlib only).

Serves an iPad-friendly (landscape) dashboard on :8765, reachable over the tailnet.
- GPU monitoring: polls nvidia-smi every 2s (util/mem/temp/power/clock). SM/TC util are marked
  n/a here (need DCGM/Nsight) — Nsight stays an explicit deeper mode, per spec.
- Metrics ingest: POST /emit {events:[{ts,gen_id,step,layer,stage,kernel,duration_ms,seq_len,...}]}
- Kernel benchmark history: POST /bench {kernel,latency_ms,...}; never overwrites the best.
- Live push via SSE (GET /events); JSON fallback GET /api/state.

Run:  CUDA-free.  python3 server.py [--port 8765] [--bind auto|0.0.0.0|<ip>]
"""
import os, sys, json, time, threading, subprocess, collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_FILE = os.path.join(HERE, "bench_history.json")
PORT = 8770          # 8765 is reserved — do not use it
FA2_BASELINE_MS = 263.1

# ---------------- state ----------------
_events = collections.deque(maxlen=8000)      # raw metric events
_timeline = collections.deque(maxlen=100)     # per-step total times
_gpus = {"ts": 0, "list": []}
_gpu_hist = collections.deque(maxlen=100)
_gen = {"gen_id": None, "start": None, "step": 0, "total_steps": 0, "layer": 0,
        "seq_len": 0, "kernel": "FA2", "dtype": "bf16"}
_lock = threading.Lock()

def _load_bench():
    if os.path.exists(BENCH_FILE):
        try:
            return json.load(open(BENCH_FILE))
        except Exception:
            pass
    return [{"kernel": "FA2", "latency_ms": FA2_BASELINE_MS, "vs_fa2": 1.0,
             "config": "SDPA FLASH bf16", "note": "baseline", "ts": time.time()}]
_bench = _load_bench()

def _save_bench():
    try:
        json.dump(_bench, open(BENCH_FILE, "w"), indent=2)
    except Exception:
        pass

# ---------------- GPU poller ----------------
def _poll_gpus():
    q = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.max.sm"
    while True:
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            lst = []
            for line in out.splitlines():
                c = [x.strip() for x in line.split(",")]
                if len(c) < 10:
                    continue
                def f(x):
                    try: return float(x)
                    except Exception: return None
                lst.append({"index": int(c[0]), "name": c[1], "util": f(c[2]),
                            "mem_used": f(c[3]), "mem_total": f(c[4]), "temp": f(c[5]),
                            "power": f(c[6]), "power_limit": f(c[7]),
                            "clock": f(c[8]), "clock_max": f(c[9]),
                            "sm_util": None, "tc_util": None})  # SM/TC need DCGM/Nsight
            with _lock:
                _gpus["ts"] = time.time(); _gpus["list"] = lst
                _gpu_hist.append({"ts": time.time(),
                                  "util": {g["index"]: g["util"] for g in lst},
                                  "temp": {g["index"]: g["temp"] for g in lst}})
        except Exception as e:
            with _lock:
                _gpus["list"] = [{"index": -1, "name": f"nvidia-smi error: {e}", "util": None}]
        time.sleep(2)

# ---------------- ingest ----------------
def ingest(events):
    with _lock:
        for ev in events:
            _events.append(ev)
            gid = ev.get("gen_id")
            if gid is not None and gid != _gen["gen_id"]:
                _gen.update({"gen_id": gid, "start": ev.get("ts", time.time()), "step": 0})
            for k in ("step", "total_steps", "layer", "seq_len", "kernel", "dtype"):
                if ev.get(k) is not None:
                    _gen[k] = ev[k]
            if ev.get("stage") in ("total_step", "step"):
                _timeline.append({"ts": ev.get("ts"), "step": ev.get("step"),
                                  "ms": ev.get("duration_ms")})

def _stage_breakdown(window_gen=True):
    """Aggregate per-stage ms + pct. Uses events of the current gen if present, else last 400 events."""
    with _lock:
        evs = list(_events)
        gid = _gen["gen_id"]
    if window_gen and gid is not None:
        evs = [e for e in evs if e.get("gen_id") == gid]
    else:
        evs = evs[-400:]
    agg = collections.OrderedDict()
    for e in evs:
        st = e.get("stage")
        if not st or st in ("total_step", "step"):
            continue
        agg[st] = agg.get(st, 0.0) + (e.get("duration_ms") or 0.0)
    tot = sum(agg.values()) or 1.0
    return [{"stage": k, "ms": round(v, 3), "pct": round(100 * v / tot, 1)}
            for k, v in sorted(agg.items(), key=lambda x: -x[1])], round(tot, 3)

def _bench_view():
    with _lock:
        recs = list(_bench)
    fastest = min(recs, key=lambda r: r["latency_ms"]) if recs else None
    current = recs[-1] if recs else None
    fa2 = next((r["latency_ms"] for r in recs if r["kernel"] == "FA2"), FA2_BASELINE_MS)
    # best latency seen per kernel name (never overwrite the historical best)
    best_per = {}
    for r in recs:
        k = r["kernel"]
        if k not in best_per or r["latency_ms"] < best_per[k]["latency_ms"]:
            best_per[k] = r
    rows = sorted(best_per.values(), key=lambda r: r["latency_ms"])
    for r in rows:
        r = r  # each row already has latency; compute vs_fa2 live
        r["vs_fa2_calc"] = round(fa2 / r["latency_ms"], 3) if r["latency_ms"] else None
    return {"rows": rows, "all": recs[-200:], "fastest": fastest, "current": current, "fa2_ms": fa2}

def snapshot():
    with _lock:
        gpus = list(_gpus["list"]); gen = dict(_gen); tl = list(_timeline)
        ghist = list(_gpu_hist)
    elapsed = (time.time() - gen["start"]) if gen["start"] else 0
    step, total = gen.get("step", 0) or 0, gen.get("total_steps", 0) or 0
    eta = (elapsed / step * (total - step)) if step and total and step <= total else None
    stages, step_total = _stage_breakdown()
    return {"ts": time.time(), "gpus": gpus, "gpu_hist": ghist,
            "overview": {**gen, "elapsed": round(elapsed, 1), "eta": round(eta, 1) if eta else None,
                         "step_total_ms": step_total},
            "stages": stages, "timeline": tl, "bench": _bench_view()}

# ---------------- HTTP ----------------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
        try: self.wfile.write(b)
        except Exception: pass
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, json.dumps(snapshot()))
        elif self.path == "/events":
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            try:
                while True:
                    self.wfile.write(f"data: {json.dumps(snapshot())}\n\n".encode()); self.wfile.flush()
                    time.sleep(1.0)
            except Exception:
                return
        else:
            self._send(404, "{}")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send(400, '{"error":"bad json"}'); return
        if self.path == "/emit":
            ingest(body.get("events", []) if isinstance(body, dict) else body)
            self._send(200, '{"ok":true}')
        elif self.path == "/bench":
            with _lock:
                _bench.append(body); _save_bench()
            self._send(200, '{"ok":true}')
        else:
            self._send(404, "{}")

def _bind_ip(pref):
    if pref and pref != "auto":
        return pref
    try:
        ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3).stdout.strip().splitlines()
        if ip and ip[0]:
            return ip[0].strip()
    except Exception:
        pass
    return "0.0.0.0"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--bind", default="auto")
    a = ap.parse_args()
    ip = _bind_ip(a.bind)
    threading.Thread(target=_poll_gpus, daemon=True).start()
    srv = ThreadingHTTPServer((ip, a.port), H)
    print(f"[h3-dashboard] serving on http://{ip}:{a.port}  (tailnet)", flush=True)
    srv.serve_forever()
