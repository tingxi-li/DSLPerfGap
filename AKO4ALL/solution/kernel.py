import torch
import torch.nn as nn

# --- Original implementation inlined below ---
import triton
import triton.language as tl
import torch

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("matmul", "triton") or {}
except Exception:
    _TUNED = {}


@triton.jit
def matmul_kernel(
    c_ptr, a_ptr, b_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_SIZE_K < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] + k * BLOCK_SIZE_K < K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    """Unified API: matmul(a, b) -> Tensor for 2D float16 matrices."""
    assert a.ndim == 2 and b.ndim == 2, "Only 2D matrices supported"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M = _TUNED.get("BLOCK_SIZE_M", 64)
    BLOCK_SIZE_N = _TUNED.get("BLOCK_SIZE_N", 64)
    BLOCK_SIZE_K = _TUNED.get("BLOCK_SIZE_K", 64)

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matmul_kernel[grid](
        c, a, b,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return c

# --- End original implementation ---


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return matmul(*args)


def get_inputs():
    a = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    return [a, b]

def get_init_inputs():
    return []
