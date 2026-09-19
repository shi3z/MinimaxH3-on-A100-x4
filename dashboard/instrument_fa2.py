# Instrumented FA2 baseline: emits per-stage CUDA-event timings + a bench record to the dashboard.
import os, sys, time
os.environ.setdefault("H3_METRICS","1")
os.environ.setdefault("H3_DASH_URL","http://100.126.237.55:8770")
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, "/mnt/ssdraid/project/h3-opt/attn_sm80")
import h3_metrics as M
from bench_attn import make_qkv, reference, bench, H, N, D

q,k,v = make_qkv()
M.set_context(gen_id=f"fa2-{int(time.time())}", kernel="FA2", dtype="bf16", seq_len=N, total_steps=50)
# simulate a 50-layer denoise step, timing attention per layer via CUDA events (async collect)
def attn(): 
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
for _ in range(3): attn()
torch.cuda.synchronize()
for layer in range(50):
    M.set_context(step=1, layer=layer)
    with M.stage("attention"):
        o = attn()
# a couple more "steps" for the timeline
for stp in range(2,5):
    t0=torch.cuda.Event(enable_timing=True); t1=torch.cuda.Event(enable_timing=True); t0.record()
    for layer in range(50):
        M.set_context(step=stp, layer=layer)
        with M.stage("attention"): attn()
    t1.record(); torch.cuda.synchronize()
    M.set_context(step=stp); M.emit("total_step", t0.elapsed_time(t1))
# measured single-layer FA2 latency -> bench history
ms = bench(attn)
M.record_bench("FA2", ms, vs_fa2=1.0, config="SDPA FLASH bf16 (instrumented)", note="baseline")
print(f"FA2 single-layer: {ms:.2f} ms; emitted stage+bench events to dashboard")
time.sleep(2)  # let the async collector flush
