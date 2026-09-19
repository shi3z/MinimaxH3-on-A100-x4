#!/usr/bin/env python3
"""H3 optimization dashboard server v2 — hierarchical performance analysis (stdlib only).

Hierarchy from real CUDA-event events (kind = step | layer | op):
  run -> step -> layer -> op(leaf).  Parent timers (_step,_layer) are totals, NEVER summed
  into the leaf-op denominator (fixes the double-count). Leaf ops + an explicit `unaccounted`
  bucket sum to the parent total.

Real GPU metrics via `nvidia-smi dmon` (sm%, mem%/HBM, pcie rx/tx, power, temp, clocks) + NVLink
throughput (counter deltas). Tensor-core util is NOT available without DCGM -> reported null.

Endpoints: GET / (html), GET /api/state, GET /events (SSE), POST /emit, POST /bench.
Loads dashboard/run_capture.jsonl on startup so a captured run is visible immediately.
"""
import os, json, time, threading, subprocess, collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_FILE = os.path.join(HERE, "bench_history.json")
RUNS_DIR = os.path.join(HERE, "runs"); os.makedirs(RUNS_DIR, exist_ok=True)
CAPTURE = os.path.join(HERE, "run_capture.jsonl")
PORT = 8770          # 8765 is reserved
FA2_BASELINE_MS = 263.1
GPU_HIST = 120       # seconds of rolling GPU samples

_lock = threading.Lock()
_runs = collections.OrderedDict()     # gen_id -> run
_cur = {"gen": None}
_gpu = {}                             # idx -> latest sample
_gpu_hist = collections.defaultdict(lambda: collections.deque(maxlen=GPU_HIST))
_nvlink_prev = {}                     # idx -> (ts, kib_total)
_ev_count = {"n": 0, "t0": time.time()}

def _new_run(gen, ev):
    return {"gen": gen, "kernel": ev.get("kernel"), "seq_len": ev.get("seq_len"),
            "dtype": ev.get("dtype"), "total_steps": ev.get("total_steps"),
            "ts_start": ev.get("ts", time.time()), "ts_last": ev.get("ts", time.time()),
            "steps": {}}

def _step(run, s):
    return run["steps"].setdefault(s, {"ms": None, "start": None, "end": None,
                                       "layers": {}, "ops": collections.defaultdict(float),
                                       "layer_ops": collections.defaultdict(lambda: collections.defaultdict(float))})

def ingest(events):
    with _lock:
        for ev in events:
            _ev_count["n"] += 1
            gen = ev.get("gen_id")
            if gen is None:
                continue
            if gen not in _runs:
                # persist the previous run for comparison before starting a new one
                if _cur["gen"] and _cur["gen"] in _runs:
                    _persist_run(_runs[_cur["gen"]])
                _runs[gen] = _new_run(gen, ev)
                while len(_runs) > 6:
                    _runs.popitem(last=False)
            _cur["gen"] = gen
            run = _runs[gen]; run["ts_last"] = ev.get("ts", time.time())
            for kf in ("kernel", "seq_len", "dtype", "total_steps"):
                if ev.get(kf) is not None:
                    run[kf] = ev[kf]
            s = ev.get("step"); kind = ev.get("kind"); dur = ev.get("duration_ms") or 0.0
            if s is None:
                continue
            st = _step(run, s)
            if kind == "step":
                st["ms"] = dur; st["end"] = ev.get("ts"); st["start"] = (ev.get("ts") or 0) - dur / 1000.0
            elif kind == "layer":
                st["layers"][ev.get("layer")] = dur
            elif kind == "op":
                op = ev.get("stage"); st["ops"][op] += dur
                st["layer_ops"][ev.get("layer")][op] += dur

def _persist_run(run):
    try:
        avg = _run_avg_step(run)
        summary = {"gen": run["gen"], "kernel": run.get("kernel"), "seq_len": run.get("seq_len"),
                   "avg_step_ms": avg, "op_avg": _run_op_avg(run), "ts": run["ts_last"]}
        json.dump(summary, open(os.path.join(RUNS_DIR, f"{run['gen']}.json"), "w"))
    except Exception:
        pass

def _run_avg_step(run):
    ms = [st["ms"] for st in run["steps"].values() if st["ms"]]
    return round(sum(ms) / len(ms), 2) if ms else None

def _run_op_avg(run):
    agg = collections.defaultdict(float); n = 0
    for st in run["steps"].values():
        if st["ms"]:
            n += 1
            for op, v in st["ops"].items():
                agg[op] += v
    return {k: round(v / n, 2) for k, v in agg.items()} if n else {}

# ---------------- GPU: nvidia-smi dmon stream + nvlink ----------------
def _dmon():
    # columns for -s pucmt: idx pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk fb bar1 ccpm rxpci txpci
    while True:
        try:
            p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "pucmt", "-d", "1"],
                                 stdout=subprocess.PIPE, text=True, bufsize=1)
            for line in p.stdout:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                c = line.split()
                if len(c) < 17:
                    continue
                def fi(x):
                    try: return float(x)
                    except Exception: return None
                idx = int(c[0]); ts = time.time()
                s = {"ts": ts, "idx": idx, "power": fi(c[1]), "gtemp": fi(c[2]), "mtemp": fi(c[3]),
                     "sm": fi(c[4]), "mem": fi(c[5]), "mclk": fi(c[10]), "pclk": fi(c[11]),
                     "fb": fi(c[12]), "bar1": fi(c[13]), "rxpci": fi(c[15]), "txpci": fi(c[16]),
                     "util": fi(c[4]), "tc": None,  # TC util needs DCGM -> unavailable
                     "nvlink": _nvlink_rate(idx, ts)}
                with _lock:
                    _gpu[idx] = s; _gpu_hist[idx].append(s)
        except Exception as e:
            with _lock:
                _gpu[-1] = {"idx": -1, "err": str(e)}
            time.sleep(3)

_nvlink_poll = {"t": 0, "val": {}}
def _nvlink_rate(idx, ts):
    # refresh nvlink counters at most every ~2s (shared across GPUs)
    if ts - _nvlink_poll["t"] > 2:
        _nvlink_poll["t"] = ts
        try:
            out = subprocess.run(["nvidia-smi", "nvlink", "-gt", "d"], capture_output=True, text=True, timeout=3).stdout
            cur = {}; g = None
            for ln in out.splitlines():
                ln = ln.strip()
                if ln.startswith("GPU "):
                    g = int(ln.split()[1].rstrip(":"))
                    cur.setdefault(g, 0)
                elif ln.startswith("Link") and g is not None and "KiB" in ln:
                    try: cur[g] += int(ln.split()[-2])
                    except Exception: pass
            for gi, tot in cur.items():
                prev = _nvlink_prev.get(gi)
                if prev:
                    dt = ts - prev[0]
                    if dt > 0:
                        _nvlink_poll["val"][gi] = max(0.0, (tot - prev[1]) / dt / 1024.0)  # MiB/s
                _nvlink_prev[gi] = (ts, tot)
        except Exception:
            pass
    return _nvlink_poll["val"].get(idx)

# ---------------- bench ----------------
def _load_bench():
    if os.path.exists(BENCH_FILE):
        try: return json.load(open(BENCH_FILE))
        except Exception: pass
    return [{"kernel": "FA2", "latency_ms": FA2_BASELINE_MS, "vs_fa2": 1.0,
             "config": "SDPA FLASH bf16", "note": "baseline", "git": "", "ts": time.time()}]
_bench = _load_bench()
def _save_bench():
    try: json.dump(_bench, open(BENCH_FILE, "w"), indent=2)
    except Exception: pass

# ---------------- snapshot / analysis ----------------
ATTN_OPS = ["attn.qkv", "attn.rope", "attn.qknorm", "attn.core", "attn.out", "attn.reshape"]

def _gpu_window(idx, t0, t1):
    with _lock:
        samples = [s for s in _gpu_hist.get(idx, []) if s["ts"] >= t0 and s["ts"] <= t1]
    return samples

def _multi_gpu_per_step(run):
    out = []
    with _lock:
        gpu_ids = sorted(_gpu.keys())
    for s in sorted(run["steps"]):
        st = run["steps"][s]
        if not st["start"] or not st["end"]:
            continue
        win = st["end"] - st["start"]
        per = {}
        eff = 0.0
        for gi in gpu_ids:
            if gi < 0: continue
            sm = [x["sm"] for x in _gpu_window(gi, st["start"], st["end"]) if x["sm"] is not None]
            m = (sum(sm) / len(sm)) if sm else 0.0
            per[gi] = round(m, 1); eff += m / 100.0
        active = [gi for gi, v in per.items() if v > 5]
        out.append({"step": s, "wall_s": round(win, 2), "per_gpu": per,
                    "effective_active": round(eff, 2), "active_count": len(active),
                    "n_gpu": len([g for g in gpu_ids if g >= 0])})
    return out

def _amdahl(step_ms, ops):
    rows = []
    for op, ms in sorted(ops.items(), key=lambda x: -x[1])[:5]:
        r = {"op": op, "ms": round(ms, 1), "pct": round(100 * ms / step_ms, 1) if step_ms else 0, "scen": {}}
        for sp in (1.1, 1.2, 1.5, 2.0):
            new = step_ms - ms * (1 - 1 / sp)
            r["scen"][sp] = {"step_ms": round(new, 1), "overall": round(step_ms / new, 3) if new else None}
        rows.append(r)
    return rows

def _warnings(run, rep_step, mg):
    w = []
    if rep_step:
        ms = rep_step["ms"]; opsum = sum(rep_step["ops"].values())
        if ms:
            un = ms - opsum
            if abs(un) / ms > 0.05:
                w.append(f"Unaccounted step time {un:.0f}ms ({100*un/ms:.1f}%) — instrumentation gap")
            if abs(un) / ms > 0.02:
                w.append(f"Parent/child mismatch >2% (step {ms:.0f}ms vs leaf-op sum {opsum:.0f}ms)")
            attn = sum(v for k, v in rep_step["ops"].items() if k.startswith("attn."))
            if attn / ms > 0.60:
                w.append(f"Attention dominates {100*attn/ms:.0f}% of step time")
    if mg:
        last = mg[-1]
        if last["active_count"] <= 1 and last["n_gpu"] > 1:
            w.append(f"Only {last['active_count']}/{last['n_gpu']} GPUs active — no multi-GPU parallelism")
        vals = [v for v in last["per_gpu"].values()]
        if vals and max(vals) > 5:
            imb = (max(vals) - (sum(vals) / len(vals))) / max(vals)
            if imb > 0.5:
                w.append(f"GPU imbalance {100*imb:.0f}% (max {max(vals):.0f}% vs mean {sum(vals)/len(vals):.0f}%)")
    return w

def _bench_view():
    with _lock: recs = list(_bench)
    fa2 = next((r["latency_ms"] for r in recs if r["kernel"] == "FA2"), FA2_BASELINE_MS)
    best_per = {}
    for r in recs:
        k = r["kernel"]
        if k not in best_per or r["latency_ms"] < best_per[k]["latency_ms"]:
            best_per[k] = dict(r)
    rows = sorted(best_per.values(), key=lambda r: r["latency_ms"])
    for r in rows:
        r["vs_fa2_calc"] = round(fa2 / r["latency_ms"], 3) if r["latency_ms"] else None
    fastest = rows[0] if rows else None
    return {"rows": rows, "all": recs[-200:], "fastest": fastest, "fa2_ms": fa2, "n": len(recs)}

def _run_payload(run):
    steps = []
    for s in sorted(run["steps"]):
        st = run["steps"][s]
        opsum = sum(st["ops"].values())
        steps.append({"step": s, "ms": round(st["ms"], 2) if st["ms"] else None,
                      "start": st["start"], "end": st["end"],
                      "layers": {int(k): round(v, 3) for k, v in st["layers"].items()},
                      "ops": {k: round(v, 3) for k, v in st["ops"].items()},
                      "layer_ops": {int(l): {k: round(v, 3) for k, v in ops.items()}
                                    for l, ops in st["layer_ops"].items()},
                      "op_sum": round(opsum, 2),
                      "unaccounted": round((st["ms"] - opsum), 2) if st["ms"] else None})
    return {"gen": run["gen"], "kernel": run.get("kernel"), "seq_len": run.get("seq_len"),
            "dtype": run.get("dtype"), "total_steps": run.get("total_steps"), "steps": steps}

def _comparisons(run):
    cur = {"avg_step_ms": _run_avg_step(run), "op_avg": _run_op_avg(run), "gen": run["gen"]}
    others = []
    for fn in sorted(os.listdir(RUNS_DIR)):
        try:
            d = json.load(open(os.path.join(RUNS_DIR, fn)))
            if d.get("gen") != run["gen"] and d.get("avg_step_ms"):
                others.append(d)
        except Exception:
            pass
    others.sort(key=lambda d: d["ts"])
    prev = others[-1] if others else None
    best = min(others + [cur], key=lambda d: d["avg_step_ms"] or 9e9) if (others or cur["avg_step_ms"]) else None
    return {"current": cur, "previous": prev, "best": best}

def snapshot():
    with _lock:
        gen = _cur["gen"]; run = _runs.get(gen)
        gpus = [dict(_gpu[i]) for i in sorted(_gpu) if i >= 0]
        hist = {i: list(_gpu_hist[i]) for i in sorted(_gpu_hist) if i >= 0}
    out = {"ts": time.time(), "gpus": gpus, "gpu_hist": hist,
           "overhead": {"events": _ev_count["n"],
                        "eps": round(_ev_count["n"] / max(time.time() - _ev_count["t0"], 1), 1),
                        "note": "CUDA-event timing, async-collected; op-level instrumentation"},
           "bench": _bench_view()}
    if not run:
        out["run"] = None; out["warnings"] = []; return out
    rp = _run_payload(run)
    # representative step = last step that has op data
    rep = None
    for s in sorted(run["steps"], reverse=True):
        if run["steps"][s]["ops"]:
            rep = run["steps"][s]; break
    mg = _multi_gpu_per_step(run)
    rep_ms = rep["ms"] if rep else None
    bottleneck = []
    if rep and rep_ms:
        for op, ms in sorted(rep["ops"].items(), key=lambda x: -x[1]):
            bottleneck.append({"op": op, "ms": round(ms, 1), "pct": round(100 * ms / rep_ms, 1)})
    out.update({"run": rp,
                "rep_step": (sorted(run["steps"])[-1] if run["steps"] else None),
                "bottleneck": bottleneck,
                "amdahl": _amdahl(rep_ms, dict(rep["ops"])) if rep else [],
                "multi_gpu": mg,
                "comparisons": _comparisons(run),
                "warnings": _warnings(run, rep, mg)})
    return out

# ---------------- HTTP ----------------
class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a): pass
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
            self.send_header("Cache-Control", "no-cache"); self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
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
        try: body = json.loads(self.rfile.read(n) or b"{}")
        except Exception: self._send(400, '{"error":"bad json"}'); return
        if self.path == "/emit":
            ingest(body.get("events", []) if isinstance(body, dict) else body); self._send(200, '{"ok":true}')
        elif self.path == "/bench":
            with _lock: _bench.append(body); _save_bench()
            self._send(200, '{"ok":true}')
        else:
            self._send(404, "{}")

def _load_capture():
    if os.path.exists(CAPTURE):
        try:
            evs = [json.loads(l) for l in open(CAPTURE) if l.strip()]
            ingest(evs)
            print(f"[h3-dashboard] loaded {len(evs)} captured events from run_capture.jsonl", flush=True)
        except Exception as e:
            print("capture load failed:", e, flush=True)

def _bind_ip(pref):
    if pref and pref != "auto": return pref
    try:
        ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3).stdout.strip().splitlines()
        if ip and ip[0]: return ip[0].strip()
    except Exception: pass
    return "0.0.0.0"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT); ap.add_argument("--bind", default="auto")
    a = ap.parse_args()
    _load_capture()
    threading.Thread(target=_dmon, daemon=True).start()
    ip = _bind_ip(a.bind)
    print(f"[h3-dashboard v2] http://{ip}:{a.port}", flush=True)
    ThreadingHTTPServer((ip, a.port), H).serve_forever()
