import os, sys
os.environ["CUDA_HOME"]="/usr/local/cuda-12.9"
os.environ["PATH"]="/usr/local/cuda-12.9/bin:"+os.environ.get("PATH","")
import torch
from torch.utils.cpp_extension import load
from bench_attn import make_qkv, reference, bench
HERE=os.path.dirname(os.path.abspath(__file__))

def build(name, defines):
    return load(name=name, sources=[os.path.join(HERE,"fa3_sm80_h3.cu")],
        extra_cuda_cflags=["-O3","-arch=sm_80","--use_fast_math",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__","-U__CUDA_NO_BFLOAT16_OPERATORS__"]+defines,
        verbose=False)

q,k,v=make_qkv(dev="cuda:0"); o=torch.empty_like(q)
ref=reference(q,k,v)
ms_fa=bench(lambda: reference(q,k,v))
print(f"FA-2 baseline: {ms_fa:.1f} ms")
variants=[]
for tag in sys.argv[1:]:
    name, defs = tag.split(":")[0], tag.split(":")[1:]
    defines=[f"-D{d}" for d in defs]
    ext=build(name, defines)
    ext.fa3(q,k,v,o); torch.cuda.synchronize()
    rel=((o.float()-ref.float()).abs().mean()/ref.float().abs().mean()).item()
    ms=bench(lambda: (lambda oo: (ext.fa3(q,k,v,oo), oo)[1])(o))
    print(f"{name:24s} rel={rel:.2e} {ms:8.1f} ms  {ms_fa/ms:.3f}x  [{' '.join(defs) or 'default'}]")
