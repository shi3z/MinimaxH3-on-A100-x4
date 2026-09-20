"""In-process ComfyUI cut generation, single-GPU OR 4-GPU Ulysses sequence-parallel.

Each rank runs the IDENTICAL cut graph (build_graph from the fleet worker) in-process via
ComfyUI's PromptExecutor with the SAME seed, so every rank reaches run_blocks in lockstep and
the Ulysses collectives (patched into MiniMaxH3Model.run_blocks + Attention by sp_runtime) match.
rank 0's SaveVideo output is the deliverable mp4; VAE decode is replicated on every rank.

Launch:
  WORLD=1:  CUDA_VISIBLE_DEVICES=4 python sp_generate.py <job.json> <tag>
  WORLD=4:  run_sp_generate.sh   (one process per GPU 4-7, NCCL)

Instrumentation (all ranks compute; rank 0 reports):
  - captures the final video LATENT (at VAEDecode input) -> logs/spgen_<tag>_latent.pt   (correctness)
  - times each run_blocks call (~= per denoise step) and VAE decode
  - locates rank 0's saved mp4
"""
import os, sys, time, json, glob
sys.path.insert(0, "/mnt/ssdraid/project/h3-opt/harness")
sys.path.insert(0, "/mnt/ssdraid/project/comfy-h3")
sys.path.insert(0, "/mnt/ssdraid/project/h3-fleet")

JOB_PATH = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "gen"
LOG = "/mnt/ssdraid/project/h3-opt/logs"
COMFY_OUT = "/mnt/ssdraid/project/comfy-h3/output"

import sp_runtime
sp_runtime.init()
RANK, WORLD = sp_runtime.RANK, sp_runtime.WORLD
import torch
import torch.distributed as dist

def log(*a):
    if RANK == 0:
        print("[spgen]", *a, flush=True)

# ---- ComfyUI in-process bootstrap (mirrors main.start_comfyui, no web server) ----
os.chdir("/mnt/ssdraid/project/comfy-h3")
import asyncio
import server, execution, nodes, folder_paths
import utils.extra_config
_cfg = os.path.join(os.getcwd(), "extra_model_paths.yaml")
if os.path.isfile(_cfg):
    utils.extra_config.load_extra_path_config(_cfg)   # register /mnt/ssd/models/comfy-h3 before node init
loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
ps = server.PromptServer(loop)
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
import comfy.model_management as mm

# install the SP seam (patches MiniMaxH3Model.run_blocks + Attention.forward; no-op math for WORLD==1)
sp_runtime.install()

# ---- timing + latent capture hooks --------------------------------------------
_run_blocks_ms = []
import comfy.ldm.minimax.model as MM
_rb = MM.MiniMaxH3Model.run_blocks
_S_seen = [None]
def _timed_run_blocks(self, h, *a, **k):
    if _S_seen[0] is None:
        _S_seen[0] = h.shape[0]
        log(f"packed seq S={h.shape[0]}  S%{WORLD}={h.shape[0]%WORLD}  (pad->{((h.shape[0]+WORLD-1)//WORLD)*WORLD})")
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record(); out = _rb(self, h, *a, **k); en.record(); torch.cuda.synchronize()
    _run_blocks_ms.append(st.elapsed_time(en)); return out
MM.MiniMaxH3Model.run_blocks = _timed_run_blocks

_vae_ms = [0.0]
_orig_vaedecode = nodes.VAEDecode.decode
def _cap_vaedecode(self, vae, samples):
    # capture the final video latent (input to VAE decode) once, rank-agnostic (all ranks identical)
    try:
        t = samples.get("samples") if isinstance(samples, dict) else samples
        torch.save({"samples": t.detach().float().cpu()}, f"{LOG}/spgen_{TAG}_latent.pt")
    except Exception as ex:
        log("latent capture failed:", ex)
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record(); r = _orig_vaedecode(self, vae, samples); en.record(); torch.cuda.synchronize()
    _vae_ms[0] += st.elapsed_time(en); return r
nodes.VAEDecode.decode = _cap_vaedecode

# ---- build the graph -----------------------------------------------------------
from comfy_worker import build_graph, WID
job = json.load(open(JOB_PATH))
g = build_graph(job)
# per-rank save prefix so the 4 ranks never collide; rank 0 is the deliverable
for nid, node in g.items():
    if node.get("class_type") == "SaveVideo":
        node["inputs"]["filename_prefix"] = f"spgen/{TAG}_r{RANK}"
prefix_sub = "spgen"

prompt_id = f"spgen_{TAG}_r{RANK}_{int(time.time())}"
valid = loop.run_until_complete(execution.validate_prompt(prompt_id, g, None))
if not valid[0]:
    log("VALIDATION FAILED:", valid[1]); sys.exit(1)

e = execution.PromptExecutor(ps, cache_type=execution.CacheType.CLASSIC,
                             cache_args={"lru": 0, "ram": 8.0, "ram_inactive": 32.0})

if WORLD > 1:
    dist.barrier()
mm_free_before = torch.cuda.mem_get_info()[0]
t_load0 = time.time()
# warm: first execute includes model load; we measure the whole thing (that's the real wall clock)
t0 = time.time()
e.execute(g, prompt_id, extra_data={}, execute_outputs=valid[2])
if WORLD > 1:
    dist.barrier()
dt = time.time() - t0
peak_gb = torch.cuda.max_memory_allocated() / 1e9

# locate rank 0's mp4
mp4 = None
cands = sorted(glob.glob(f"{COMFY_OUT}/{prefix_sub}/{TAG}_r{RANK}*.mp4"), key=os.path.getmtime, reverse=True)
if cands:
    mp4 = cands[0]

if WORLD > 1:
    pm = torch.tensor([peak_gb], device="cuda:0"); dist.all_reduce(pm, op=dist.ReduceOp.MAX); peak_gb = pm.item()

if RANK == 0:
    steps = _run_blocks_ms
    # exclude step 0 (may include lazy alloc); report steady median if enough steps
    body = steps[1:] if len(steps) > 2 else steps
    import statistics
    med = statistics.median(body) if body else 0.0
    result = {
        "tag": TAG, "world": WORLD, "seed": job["seed"], "num_frames": job["num_frames"],
        "wall_s": round(dt, 2), "run_blocks_calls": len(steps),
        "run_blocks_ms_each": [round(x, 1) for x in steps],
        "run_blocks_ms_median": round(med, 1),
        "vae_decode_ms": round(_vae_ms[0], 1),
        "peak_mem_gb": round(peak_gb, 2), "mp4": mp4,
        "latent_pt": f"{LOG}/spgen_{TAG}_latent.pt",
    }
    json.dump(result, open(f"{LOG}/spgen_{TAG}_result.json", "w"), indent=1)
    log("=== RESULT ===")
    for k, v in result.items():
        if k != "run_blocks_ms_each":
            log(f"  {k}: {v}")
    log(f"  run_blocks_ms_each: {result['run_blocks_ms_each']}")

if WORLD > 1:
    dist.barrier()
    dist.destroy_process_group()
