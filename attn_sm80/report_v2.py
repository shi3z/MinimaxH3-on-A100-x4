import os, sys, subprocess, statistics, time
os.environ["CUDA_HOME"]="/usr/local/cuda-12.9"
os.environ["PATH"]="/usr/local/cuda-12.9/bin:"+os.environ.get("PATH","")
os.environ["H3_METRICS"]="1"; os.environ.setdefault("H3_DASH_URL","http://100.126.237.55:8770")
sys.path.insert(0,"/mnt/ssdraid/project/h3-opt/dashboard")
sys.path.insert(0,"/mnt/ssdraid/project/h3-opt/attn_sm80")
import torch
from torch.utils.cpp_extension import load
from bench_attn import reference
HERE="/mnt/ssdraid/project/h3-opt/attn_sm80"
try: import h3_metrics as M
except Exception as e: M=None; print("no metrics",e)
def git():
    try: return subprocess.check_output(["git","-C",HERE,"rev-parse","--short","HEAD"],text=True).strip()
    except Exception: return "n/a"
def build(name,defs):
    return load(name=name,sources=[os.path.join(HERE,"fa3_sm80_h3.cu")],
      extra_cuda_cflags=["-O3","-arch=sm_80","--use_fast_math",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__","-U__CUDA_NO_BFLOAT16_OPERATORS__"]+defs,verbose=False)
def med(fn,n=20,w=5):
    for _ in range(w): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        t=time.time(); fn(); torch.cuda.synchronize(); ts.append((time.time()-t)*1000)
    return statistics.median(ts)

# best v2 config found: BN=32 STAGES=2 MINCTA=3 (3 CTA)
BEST=["-DBM=64","-DBN=32","-DSTAGES=2","-DMINCTA=3","-DPAD=8"]
CFG="BM=64 BN=32 STAGES=2 warps=4 MINCTA=3 PAD=8 pipelined"
ANCHOR={40868:263.0, 46052:342.1}
ext=build("fa3_v2_clean", BEST)
at=ext.pp_attrs().tolist(); regs,smem_b,maxblk,clk_khz,occ,_=at
for N in [40868,46052]:
    H,Dd=56,128; g=torch.Generator(device="cuda:0").manual_seed(0)
    q=torch.randn(1,H,N,Dd,generator=g,device="cuda:0",dtype=torch.bfloat16)
    k=torch.randn(1,H,N,Dd,generator=g,device="cuda:0",dtype=torch.bfloat16)
    v=torch.randn(1,H,N,Dd,generator=g,device="cuda:0",dtype=torch.bfloat16)
    o=torch.empty_like(q); ref=reference(q,k,v)
    ext.fa3_pp(q,k,v,o); torch.cuda.synchronize()
    rel=((o.float()-ref.float()).abs().mean()/ref.float().abs().mean()).item()
    fa2=med(lambda: reference(q,k,v)); v2=med(lambda: ext.fa3_pp(q,k,v,o))
    anc=ANCHOR[N]
    print(f"N={N}: FA2(meas)={fa2:.1f}  v2(CLEAN)={v2:.1f} ms  vs_anchor({anc})={anc/v2:.3f}x  rel={rel:.2e}  "
          f"regs={regs:.0f} smem={smem_b/1024:.1f}KB occ={occ:.3f}")
    if M:
        step=22891-50*(anc-v2)
        M.set_context(kernel="FA3_SM80_H3_v2", seq_len=N)
        M.record_bench("FA3_SM80_H3_v2", v2, vs_fa2=anc/v2, config=CFG, step_ms=step,
            correctness=rel, regs=int(regs), smem_kb=smem_b/1024, occupancy=occ,
            git=git(), seq_len=N, dtype="bf16")

# PROFILE substages at N=40868
extp=build("fa3_v2_prof", BEST+["-DPROFILE"])
N=40868; H,Dd=56,128; g=torch.Generator(device="cuda:0").manual_seed(0)
q=torch.randn(1,H,N,Dd,generator=g,device="cuda:0",dtype=torch.bfloat16)
k=torch.randn(1,H,N,Dd,generator=g,device="cuda:0",dtype=torch.bfloat16)
v=torch.randn(1,H,N,Dd,generator=g,device="cuda:0",dtype=torch.bfloat16)
o=torch.empty_like(q)
extp.reset_prof(); extp.fa3_pp(q,k,v,o); torch.cuda.synchronize()
p=extp.get_prof().tolist()[:4]  # wait, qk, softmax, pv
ghz=clk_khz/1e6; tot=sum(p) or 1
names=["cp_async_wait","qk_gemm","softmax","pv_gemm"]
print("v2 PROFILE substages (clock64, %):")
v2_clean=med(lambda: extp.fa3_pp(q,k,v,o))  # note: profile build, not for latency
# use the clean v2 number for scale
clean=None
extc=build("fa3_v2_clean", BEST); clean=med(lambda: extc.fa3_pp(q,k,v,o))
for nm,c in zip(names,p):
    frac=c/tot; print(f"  {nm:16s}: {100*frac:5.1f}%   {frac*clean:.1f} ms")
    if M:
        try: M.emit(nm, frac*clean)
        except Exception as e: print("emit fail",e)
if M: time.sleep(1.0); print("posted v2 to dashboard")
