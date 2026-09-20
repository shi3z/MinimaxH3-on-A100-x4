"""Measure FA2 + FA3_SM80_H3, get attrs, profile sub-stages, post to dashboard + print table."""
import os, sys, subprocess, statistics, time
os.environ["CUDA_HOME"]="/usr/local/cuda-12.9"
os.environ["PATH"]="/usr/local/cuda-12.9/bin:"+os.environ.get("PATH","")
os.environ["H3_METRICS"]="1"
os.environ["H3_DASH_URL"]=os.environ.get("H3_DASH_URL","http://100.126.237.55:8770")
sys.path.insert(0,"/mnt/ssdraid/project/h3-opt/dashboard")
sys.path.insert(0,"/mnt/ssdraid/project/h3-opt/attn_sm80")
import torch
from torch.utils.cpp_extension import load
from bench_attn import make_qkv, reference, verify
HERE="/mnt/ssdraid/project/h3-opt/attn_sm80"
try:
    import h3_metrics as M
except Exception as ex:
    M=None; print("h3_metrics unavailable:", ex)

def git():
    try: return subprocess.check_output(["git","-C",HERE,"rev-parse","--short","HEAD"],text=True).strip()
    except Exception: return "n/a"

def build(name, defines):
    return load(name=name, sources=[os.path.join(HERE,"fa3_sm80_h3.cu")],
        extra_cuda_cflags=["-O3","-arch=sm_80","--use_fast_math",
          "-U__CUDA_NO_BFLOAT16_CONVERSIONS__","-U__CUDA_NO_BFLOAT16_OPERATORS__"]+defines, verbose=False)

def med(fn, n=20, w=5):
    for _ in range(w): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        t=time.time(); fn(); torch.cuda.synchronize(); ts.append((time.time()-t)*1000)
    return statistics.median(ts)

BEST=["-DBM=64","-DBN=64","-DSTAGES=1","-DMINCTA=3","-DPAD=8"]
CFG="BM=64 BN=64 STAGES=1 warps=4 MINCTA=3 PAD=8"
N=int(os.environ.get("H3_N","46052")); LAYERS=50; FA2_ANCHOR=342.1
H,Dd=56,128
g=torch.Generator(device="cuda:0").manual_seed(0)
q=torch.randn(1,H,N,Dd,generator=g,device="cuda:0",dtype=torch.bfloat16)
k=torch.randn(1,H,N,Dd,generator=g,device="cuda:0",dtype=torch.bfloat16)
v=torch.randn(1,H,N,Dd,generator=g,device="cuda:0",dtype=torch.bfloat16)
o=torch.empty_like(q)
ref=reference(q,k,v)

# clean perf build
ext=build("fa3_clean", BEST)
rel=((ext.fa3(q,k,v,o),o)[1].float().sub(ref.float()).abs().mean()/ref.float().abs().mean()).item()
fa2=med(lambda: reference(q,k,v))
v0=med(lambda: ext.fa3(q,k,v,o))
at=ext.kernel_attrs().tolist()
regs,smem_b,maxblk,clk_khz,occ,nthr=at
print(f"\nS={N}  (FA2 anchor from coordinator = {FA2_ANCHOR} ms/layer)")
print(f"FA-2 baseline (SDPA-FLASH, measured here) : {fa2:.1f} ms")
print(f"FA3_SM80_H3 v0 (CLEAN)   : {v0:.1f} ms  (vs measured {fa2/v0:.3f}x, vs anchor {FA2_ANCHOR/v0:.3f}x)  rel={rel:.2e}")
print(f"regs={regs:.0f} smem={smem_b/1024:.1f}KB maxblocks/SM={maxblk:.0f} occ={occ:.3f} smclk={clk_khz/1e6:.3f}GHz")
step_proj = 22891 - LAYERS*(FA2_ANCHOR - v0)
print(f"step ms (proj, 50 layers): {step_proj:.0f} ms   [22891 - 50*({FA2_ANCHOR}-{v0:.1f})]")

# PROFILE build for sub-stage cycles
extp=build("fa3_profile", BEST+["-DPROFILE"])
extp.reset_prof(); extp.fa3(q,k,v,o); torch.cuda.synchronize()
p=extp.get_prof().tolist()[:4]   # wait, qk, softmax, pv  (cycles, CTA0 summed over tiles)
ghz=clk_khz/1e6
ph_ms=[c/(ghz*1e9)*1e3 for c in p]   # cycles -> ms (per-CTA aggregate along its tile loop)
names=["cp_async_wait","qk_gemm","softmax","pv_gemm"]
print("\nPer-CTA sub-stage (PROFILE build, clock64; perturbs timing):")
tot=sum(ph_ms)
for nm,ms in zip(names,ph_ms):
    print(f"  {nm:16s}: {ms*1000:8.3f} us/CTA-loop  ({100*ms/tot:5.1f}%)")

if M:
    M.set_context(kernel="FA3_SM80_H3_v0", seq_len=N)
    M.record_bench("FA2_SDPA_FLASH", fa2, vs_fa2=1.0, config="cuDNN/flash ref",
        step_ms=22891, correctness=0.0, seq_len=N, dtype="bf16", git=git())
    M.record_bench("FA3_SM80_H3_v0", v0, vs_fa2=FA2_ANCHOR/v0, config=CFG,
        step_ms=step_proj, correctness=rel, regs=int(regs), smem_kb=smem_b/1024,
        occupancy=occ, git=git(), seq_len=N, dtype="bf16")
    # emit sub-stage ms (fraction of clean latency, so dashboard ATTENTION DETAIL is on real timescale)
    frac=[x/tot for x in ph_ms]
    emit={"cp_async_wait":frac[0]*v0,"qk_gemm":frac[1]*v0,"softmax":frac[2]*v0,"pv_gemm":frac[3]*v0}
    for nm,ms in emit.items():
        try: M.emit(nm, ms)
        except Exception as ex: print("emit fail", nm, ex)
    time.sleep(1.0)
    print("\nposted to dashboard", os.environ["H3_DASH_URL"])
