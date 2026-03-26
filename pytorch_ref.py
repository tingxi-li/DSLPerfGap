"""kernels/gemm/pytorch_ref.py — reference GEMM using PyTorch."""
import torch

def run(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Standard matrix multiply: C = A @ B, accumulated in float32."""
    return torch.mm(A.float(), B.float()).to(A.dtype)
