"""Correctness + latency harness for FA3_SM80_H3, hard-wired to the MiniMax-H3 geometry.

A candidate kernel is any callable f(q,k,v)->o with q,k,v,o = [B,H,N,D] bf16, non-causal.
Reference = PyTorch SDPA forced to the FlashAttention-2 backend (the production path).

Usage:  from bench_attn import H3_SHAPE, make_qkv, reference, verify, bench
"""
import time, torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# fixed H3 geometry (see DESIGN.md). N is the real captured packed length.
B, H, N, D = 1, 56, 40868, 128
H3_SHAPE = (B, H, N, D)

def make_qkv(seed=0, dev="cuda:0", dtype=torch.bfloat16):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(B, H, N, D, generator=g, device=dev, dtype=dtype)
    k = torch.randn(B, H, N, D, generator=g, device=dev, dtype=dtype)
    v = torch.randn(B, H, N, D, generator=g, device=dev, dtype=dtype)
    return q, k, v

def reference(q, k, v):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

def verify(fn, q, k, v, tol=5e-3):
    ref = reference(q, k, v)
    o = fn(q, k, v)
    err = (o.float() - ref.float()).abs()
    rel = (err.mean() / (ref.float().abs().mean() + 1e-8)).item()
    ok = rel < tol and torch.isfinite(o).all().item()
    print(f"  correctness: max={err.max().item():.3e} rel_mean={rel:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok

def bench(fn, n=8, w=3):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): o = fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000.0

if __name__ == "__main__":
    q, k, v = make_qkv()
    ms_fa = bench(lambda: reference(q, k, v))
    print(f"FA-2 (SDPA FLASH) baseline: {ms_fa:.2f} ms/layer  [target to beat]")
    print(f"geometry: B{B} H{H} N{N} D{D} bf16 non-causal, A100 sm_80")
