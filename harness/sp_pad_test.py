"""Uneven-S padding correctness test (4-rank, GPU4-7).

Forces a packed length S NOT divisible by WORLD to exercise sp_runtime's pad path
(pad tail -> shard -> run 50 blocks -> drop pad keys in attention -> all-gather -> drop pad).
Compares SP real-token output to the single-GPU full-sequence reference (same real tokens).
Reuses the real model weights + captured block-loop tensors, truncated to an odd length.

Run: harness/run_sp_generate.sh is for gen; this uses the same 4-rank env:
  for r in 0..3: CUDA_VISIBLE_DEVICES=$((4+r)) RANK=$r WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=P python sp_pad_test.py
"""
import os, sys, torch
sys.path.insert(0, "/mnt/ssdraid/project/h3-opt/harness")
sys.path.insert(0, "/mnt/ssdraid/project/comfy-h3")
import sp_runtime
sp_runtime.init()
RANK, WORLD = sp_runtime.RANK, sp_runtime.WORLD
import torch.distributed as dist
LOG = "/mnt/ssdraid/project/h3-opt/logs"
UNET = "/mnt/ssd/models/comfy-h3/diffusion_models/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"

import comfy.sd, comfy.model_management as mm
mp = comfy.sd.load_diffusion_model(UNET, model_options={}); mm.load_models_gpu([mp])
dm = mp.model.diffusion_model
sp_runtime.install()

gi = torch.load(f"{LOG}/io_blockloop_in.pt", map_location="cpu")
S_full = gi["h_in"].shape[0]

def clamp_segs(segs, S):
    out = []
    for a, b, row in segs:
        if a >= S: continue
        out.append((a, min(b, S), row))
    return out

def run_case(S_test):
    h_in = gi["h_in"][:S_test].to("cuda:0").contiguous()
    t_emb = gi["t_emb"].to("cuda:0")
    rope = gi["rope_freqs"][:, :S_test].to("cuda:0").contiguous()
    segs = clamp_segs(gi["mod_segments"], S_test)
    # single-GPU reference: run the 50 blocks directly on the FULL sequence (SP attention gate
    # is off outside run_blocks, so this is the original full-seq flash path). Identical on all ranks.
    with torch.no_grad():
        hr = h_in.clone()
        for blk in dm.blocks:
            hr = blk(hr, t_emb, segs, rope, {})
    # SP path: patched run_blocks pads S_test up to a multiple of WORLD internally
    with torch.no_grad():
        out = dm.run_blocks(h_in.clone(), t_emb, segs, rope, {})
    if RANK == 0:
        err = (out.float() - hr.float()).abs()
        Spad = ((S_test + WORLD - 1) // WORLD) * WORLD
        print(f"[pad-test] S={S_test} S%{WORLD}={S_test%WORLD} pad->{Spad} (P={Spad-S_test})  "
              f"real-token max_err={err.max().item():.3e} mean={err.mean().item():.3e}  "
              f"ref_absmax={hr.abs().max().item():.3f}", flush=True)

# exercise several remainders: P=1,2,3 and the exact-divisible fast path
for S_test in [S_full - 1, S_full - 2, S_full - 3, S_full]:
    run_case(S_test)
    dist.barrier()

dist.destroy_process_group()
