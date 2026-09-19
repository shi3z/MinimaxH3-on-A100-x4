import os, sys, torch, time
sys.path.insert(0, "/mnt/ssdraid/project/h3-opt/harness")
sys.path.insert(0, "/mnt/ssdraid/project/comfy-h3")
import sp_runtime
sp_runtime.init()
RANK, WORLD = sp_runtime.RANK, sp_runtime.WORLD
UNET = "/mnt/ssd/models/comfy-h3/diffusion_models/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
LOG = "/mnt/ssdraid/project/h3-opt/logs"
import comfy.sd, comfy.model_management as mm
mp = comfy.sd.load_diffusion_model(UNET, model_options={}); mm.load_models_gpu([mp])
dm = mp.model.diffusion_model
sp_runtime.install()
gi = torch.load(f"{LOG}/io_blockloop_in.pt", map_location="cpu")
go = torch.load(f"{LOG}/io_blockloop_out.pt", map_location="cpu")
h_in = gi["h_in"].to("cuda:0"); t_emb = gi["t_emb"].to("cuda:0")
rope = gi["rope_freqs"].to("cuda:0"); segs = gi["mod_segments"]; h_ref = go["h_out"].to("cuda:0")
import torch.distributed as dist
with torch.no_grad():
    out = dm.run_blocks(h_in, t_emb, segs, rope, {})     # SP via the production seam
if RANK == 0:
    err = (out.float() - h_ref.float()).abs()
    print(f"[seam SP verify WORLD={WORLD}] max_err={err.max().item():.4e} mean_err={err.mean().item():.4e}", flush=True)
# bench through the seam
for _ in range(2): dm.run_blocks(h_in, t_emb, segs, rope, {})
torch.cuda.synchronize(); dist.barrier()
st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True); st.record()
for _ in range(6): dm.run_blocks(h_in, t_emb, segs, rope, {})
en.record(); torch.cuda.synchronize(); ms = st.elapsed_time(en)/6
tt = torch.tensor([ms], device="cuda:0"); dist.all_reduce(tt, op=dist.ReduceOp.MAX)
if RANK == 0: print(f"[seam SP bench WORLD={WORLD}] run_blocks: {tt.item():.1f} ms  (1-GPU baseline 18661 ms)", flush=True)
dist.destroy_process_group()
