"""FA3_SM80_H3 backend + build. Switchable: attn(q,k,v, backend='fa3_sm80_h3'|'sdpa_flash')."""
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.9"
os.environ["PATH"] = "/usr/local/cuda-12.9/bin:" + os.environ.get("PATH", "")
import torch
from torch.utils.cpp_extension import load
from torch.nn.attention import SDPBackend, sdpa_kernel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None

def _build():
    global _ext
    if _ext is None:
        _ext = load(
            name="fa3_sm80_h3",
            sources=[os.path.join(_HERE, "fa3_sm80_h3.cu")],
            extra_cuda_cflags=["-O3", "-arch=sm_80", "--use_fast_math",
                               "-DSTAGES=1", "-DMINCTA=3", "-DPAD=8",
                               "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                               "-U__CUDA_NO_BFLOAT16_OPERATORS__"],
            verbose=True,
        )
    return _ext

def fa3_sm80_h3(q, k, v):
    ext = _build()
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    o = torch.empty_like(q)
    ext.fa3(q, k, v, o)
    return o

def sdpa_flash(q, k, v):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

def attn(q, k, v, backend="fa3_sm80_h3"):
    if backend == "fa3_sm80_h3":
        return fa3_sm80_h3(q, k, v)
    elif backend == "sdpa_flash":
        return sdpa_flash(q, k, v)
    raise ValueError(backend)
