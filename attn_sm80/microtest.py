import os
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
import torch
from attn_sm80 import _build

ext = _build()
dev = "cuda:0"
torch.manual_seed(0)

# ---- QK test: S[16,64] = Q[16,128] @ K[64,128]^T ----
Q = torch.randn(16, 128, device=dev, dtype=torch.bfloat16)
K = torch.randn(64, 128, device=dev, dtype=torch.bfloat16)
S = torch.zeros(16, 64, device=dev, dtype=torch.float32)
ext.test_qk(Q, K, S)
ref = (Q.float() @ K.float().t())
err = (S - ref).abs()
print(f"[QK]  max_err={err.max().item():.3e} rel={ (err.mean()/ref.abs().mean()).item():.3e}")

# ---- PV test: O[16,128] = P[16,64] @ V[64,128] ----
P = torch.rand(16, 64, device=dev, dtype=torch.bfloat16)
V = torch.randn(64, 128, device=dev, dtype=torch.bfloat16)
O = torch.zeros(16, 128, device=dev, dtype=torch.float32)
ext.test_pv(P, V, O)
refo = (P.float() @ V.float())
erro = (O - refo).abs()
print(f"[PV]  max_err={erro.max().item():.3e} rel={ (erro.mean()/refo.abs().mean()).item():.3e}")
