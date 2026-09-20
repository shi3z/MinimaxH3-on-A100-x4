"""Full-H3 4-GPU integration measurement (sequence-sharded across GPU4-7).

Runs the production run_blocks seam under sp_runtime (shard ONCE, all 50 blocks local on the
shard, Ulysses all-to-all only inside attention, all-gather ONCE) for multiple steps, and reports
the per-op breakdown, step latency, correctness, and memory. This is the real distributed
transformer per denoise step (run_blocks = ~97% of the H3 step).

Run: harness/run_full_sp.sh   (one process per GPU 4-7, NCCL)
"""
import os, sys, statistics, torch
sys.path.insert(0, "/mnt/ssdraid/project/h3-opt/harness")
sys.path.insert(0, "/mnt/ssdraid/project/comfy-h3")
import sp_runtime
sp_runtime.init()
RANK, WORLD = sp_runtime.RANK, sp_runtime.WORLD
import torch.distributed as dist
UNET = "/mnt/ssd/models/comfy-h3/diffusion_models/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
LOG = "/mnt/ssdraid/project/h3-opt/logs"
PREP_FINAL_MS = 340.0   # measured embed + final-layer overhead per step (replicated, outside run_blocks)
BASE_1GPU_BLOCKS_MS = 18661.0

import comfy.sd, comfy.model_management as mm
mp = comfy.sd.load_diffusion_model(UNET, model_options={}); mm.load_models_gpu([mp])
dm = mp.model.diffusion_model
sp_runtime.install()

gi = torch.load(f"{LOG}/io_blockloop_in.pt", map_location="cpu")
go = torch.load(f"{LOG}/io_blockloop_out.pt", map_location="cpu")
h_in = gi["h_in"].to("cuda:0"); t_emb = gi["t_emb"].to("cuda:0")
rope = gi["rope_freqs"].to("cuda:0"); segs = gi["mod_segments"]; h_ref = go["h_out"].to("cuda:0")
S = h_in.shape[0]

def run():
    return dm.run_blocks(h_in, t_emb, segs, rope, {})

# correctness (once)
with torch.no_grad():
    out = run()
if RANK == 0:
    err = (out.float() - h_ref.float()).abs()
    print(f"[correctness] 4-GPU full block-loop vs 1-GPU ground truth: max_err={err.max():.3e} mean={err.mean():.3e}", flush=True)

# warmup
for _ in range(3): run()
torch.cuda.synchronize(); dist.barrier()
torch.cuda.reset_peak_memory_stats()

NSTEP = 6
step_ms = []; cats = []
for i in range(NSTEP):
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    dist.barrier(); st.record()
    run()
    en.record(); torch.cuda.synchronize()
    step_ms.append(st.elapsed_time(en))
    cats.append(sp_runtime.get_timing())

peak_gb = torch.cuda.max_memory_allocated() / 1e9
# reduce peak mem across ranks (report max)
pm = torch.tensor([peak_gb], device="cuda:0"); dist.all_reduce(pm, op=dist.ReduceOp.MAX)

# steady-state = exclude the first measured step
steady = step_ms[1:]
sm_med = statistics.median(steady)
# category medians across steady steps (rank-local); all_reduce MAX for straggler view
catkeys = ["norm_mod","qkv","a2a_fwd","fa2_local","a2a_inv","out_proj","residual","ffn","gather"]
catmed = {}
for k in catkeys:
    vals = [c.get(k,0.0) for c in cats[1:]]
    catmed[k] = statistics.median(vals) if vals else 0.0
# reduce category medians (MAX across ranks)
ct = torch.tensor([catmed[k] for k in catkeys], device="cuda:0")
dist.all_reduce(ct, op=dist.ReduceOp.MAX)

if RANK == 0:
    cm = {k: ct[i].item() for i, k in enumerate(catkeys)}
    attn = cm["a2a_fwd"] + cm["fa2_local"] + cm["a2a_inv"]
    comm = cm["a2a_fwd"] + cm["a2a_inv"] + cm["gather"]
    nonattn = cm["norm_mod"] + cm["qkv"] + cm["out_proj"] + cm["residual"] + cm["ffn"]
    tot = sum(cm.values())
    full_step = sm_med + PREP_FINAL_MS
    print(f"\n=== 4-GPU FULL H3 (sequence-sharded, GPU4-7, S={S}, steady s1+) ===")
    print(f"run_blocks/step (median):   {sm_med:8.1f} ms   (1-GPU baseline {BASE_1GPU_BLOCKS_MS:.0f} ms -> {BASE_1GPU_BLOCKS_MS/sm_med:.2f}x)")
    print(f"full step (+{PREP_FINAL_MS:.0f}ms prep/final): {full_step:8.1f} ms = {full_step/1000:.2f} s   (1-GPU ~22891 ms -> {22891/full_step:.2f}x)")
    print(f"peak mem/GPU (max rank):    {pm.item():8.2f} GB")
    print(f"per-op / step (summed over 50 layers, ms, MAX across ranks):")
    for k in catkeys:
        print(f"    {k:10s} {cm[k]:8.1f}   ({100*cm[k]/tot:5.1f}%)")
    print(f"  -> attention (a2a_fwd+fa2_local+a2a_inv): {attn:8.1f} ms ({100*attn/tot:.1f}%)")
    print(f"  -> communication (all2all+gather):        {comm:8.1f} ms ({100*comm/tot:.1f}%)")
    print(f"  -> non-attention local (norm/qkv/out/residual/ffn): {nonattn:8.1f} ms ({100*nonattn/tot:.1f}%)")
    print(f"  -> measured/unaccounted vs run_blocks: {sm_med - tot:8.1f} ms")
    print(f"all steady step_ms: {[round(x,1) for x in steady]}")
dist.destroy_process_group()
