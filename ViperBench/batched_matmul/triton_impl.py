import torch
import triton
import triton.language as tl

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("batched_matmul", "triton") or {}
except Exception:
    _TUNED = {}

@triton.jit
def batched_vecmat_kernel(
        A,  # shape: [dim_m, dim_k]
        B,  # shape: [dim_m, dim_n, dim_k]
        dim_m, dim_n, dim_k,
        output,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)
    output_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_n \
        + (n_index * block_n + tl.arange(0, block_n))[None, :]

    vecmat = tl.zeros([block_m, block_n], dtype=A.dtype.element_ty)
    k_blocks = dim_k // block_k
    for k_index in range(k_blocks):
        a_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, :]
        a = tl.load(A + a_tile)

        b_tile = (m_index * block_m + tl.arange(0, block_m))[None, :, None] * dim_n * dim_k \
            + (n_index * block_n + tl.arange(0, block_n))[:, None, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, None, :]
        b = tl.load(B + b_tile)

        expanded_a, _ = tl.broadcast(a, b)
        vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))

    tl.store(output + output_tile, vecmat)


def batched_matmul(A, B):
    """
    Batched vector-matrix product using Triton.
    A: [M, K], B: [M, N, K]
    Output: [M, N]
    """
    block_m = _TUNED.get("block_m", 16)
    block_n = _TUNED.get("block_n", 32)
    block_k = _TUNED.get("block_k", 64)
    num_warps = 4
    num_stages = 1

    M, K = A.shape
    _, N, _ = B.shape

    # Pad dimensions to be divisible by block sizes
    M_pad = triton.cdiv(M, block_m) * block_m
    N_pad = triton.cdiv(N, block_n) * block_n
    K_pad = triton.cdiv(K, block_k) * block_k

    if M_pad != M or K_pad != K:
        A_padded = torch.zeros(M_pad, K_pad, device=A.device, dtype=A.dtype)
        A_padded[:M, :K] = A
    else:
        A_padded = A

    if M_pad != M or N_pad != N or K_pad != K:
        B_padded = torch.zeros(M_pad, N_pad, K_pad, device=B.device, dtype=B.dtype)
        B_padded[:M, :N, :K] = B
    else:
        B_padded = B

    output = torch.zeros(M_pad, N_pad, device=A.device, dtype=A.dtype)

    grid = (M_pad // block_m, N_pad // block_n)
    batched_vecmat_kernel[grid](
        A_padded, B_padded, M_pad, N_pad, K_pad, output,
        block_m=block_m, block_n=block_n, block_k=block_k,
        num_warps=num_warps, num_stages=num_stages
    )
    return output[:M, :N]
