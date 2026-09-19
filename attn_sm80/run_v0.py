import os
os.environ["CUDA_HOME"]="/usr/local/cuda-12.9"
import torch
from bench_attn import make_qkv, verify, bench
from attn_sm80 import attn

q,k,v = make_qkv(dev="cuda:0")
print("shapes", q.shape)

# correctness
ok = verify(lambda a,b,c: attn(a,b,c,"fa3_sm80_h3"), q,k,v)

# baseline + v0 timing
ms_fa = bench(lambda: attn(q,k,v,"sdpa_flash"))
ms_v0 = bench(lambda: attn(q,k,v,"fa3_sm80_h3"))
print(f"FA-2 baseline : {ms_fa:.2f} ms")
print(f"v0 fa3_sm80_h3: {ms_v0:.2f} ms   ({ms_fa/ms_v0:.3f}x vs FA-2)")
