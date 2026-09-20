import os, sys
os.environ["CUDA_HOME"]="/usr/local/cuda-12.9"
os.environ["PATH"]="/usr/local/cuda-12.9/bin:"+os.environ.get("PATH","")
import torch, statistics, time
from torch.utils.cpp_extension import load
import torch.nn.functional as Fnn
from bench_attn import make_qkv, reference, bench
HERE=os.path.dirname(os.path.abspath(__file__))

def build(name, defines):
    return load(name=name, sources=[os.path.join(HERE,"fa3_sm80_h3.cu")],
        extra_cuda_cflags=["-O3","-arch=sm_80","--use_fast_math",
          "-U__CUDA_NO_BFLOAT16_CONVERSIONS__","-U__CUDA_NO_BFLOAT16_OPERATORS__"]+defines, verbose=False)

def med(fn,n=15,w=5):
    for _ in range(w): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        t=time.time(); fn(); torch.cuda.synchronize(); ts.append((time.time()-t)*1000)
    return statistics.median(ts)

# small-N correctness first
dev="cuda:0"
q,k,v=make_qkv(dev=dev); o=torch.empty_like(q)
ref=reference(q,k,v)
ms_fa=med(lambda: reference(q,k,v))
print(f"FA-2 baseline: {ms_fa:.1f} ms  (N={q.shape[2]})")
for tag in sys.argv[1:]:
    parts=tag.split(":"); name=parts[0]; defs=parts[1:]
    ext=build(name,[f"-D{d}" for d in defs])
    # small-N correctness
    okall=True
    for Ns in [128,192,256]:
        H=2
        qa=torch.randn(1,H,Ns,128,device=dev,dtype=torch.bfloat16)
        ka=torch.randn(1,H,Ns,128,device=dev,dtype=torch.bfloat16)
        va=torch.randn(1,H,Ns,128,device=dev,dtype=torch.bfloat16)
        oa=torch.empty_like(qa); ext.fa3_pp(qa,ka,va,oa)
        rf=Fnn.scaled_dot_product_attention(qa,ka,va,is_causal=False)
        rel=((oa.float()-rf.float()).abs().mean()/rf.float().abs().mean()).item()
        if rel>5e-3: okall=False
    ext.fa3_pp(q,k,v,o); torch.cuda.synchronize()
    rel=((o.float()-ref.float()).abs().mean()/ref.float().abs().mean()).item()
    ms=med(lambda: (ext.fa3_pp(q,k,v,o),None)[1] or 0 and None or ext.fa3_pp(q,k,v,o))
    ms=med(lambda: ext.fa3_pp(q,k,v,o))
    print(f"{name:22s} rel={rel:.2e} small_ok={okall}  {ms:8.1f} ms  {ms_fa/ms:.3f}x  [{' '.join(defs)}]")
