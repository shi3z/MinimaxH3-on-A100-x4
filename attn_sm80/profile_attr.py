import os
os.environ["CUDA_HOME"]="/usr/local/cuda-12.9"
import torch, ctypes
from attn_sm80 import _build, attn, sdpa_flash
from bench_attn import make_qkv, bench
ext=_build()

# time-bracket the kernel and query attributes via cudaOccupancy isn't exposed;
# use torch.cuda for regs indirectly. We report smem and grid analytically.
q,k,v=make_qkv(dev="cuda:0")
# warmup+run to make sure module loaded
o=attn(q,k,v,"fa3_sm80_h3"); torch.cuda.synchronize()

# theoretical: BM=64,BN=64,D=128 -> smem=(64*128+64*128+64*128+64*64)*2 bytes
smem=(64*128*3+64*64)*2
print("smem/CTA bytes:", smem, "KB:", smem/1024)
print("grid CTAs:", (40868+63)//64, "*56 =", ((40868+63)//64)*56)
props=torch.cuda.get_device_properties(0)
print("SMs:", props.multi_processor_count, "smem/SM:", props.shared_memory_per_multiprocessor if hasattr(props,'shared_memory_per_multiprocessor') else 'n/a')
print("regs/SM:", getattr(props,'regs_per_multiprocessor','n/a'), "max_threads/SM:", props.max_threads_per_multi_processor)
