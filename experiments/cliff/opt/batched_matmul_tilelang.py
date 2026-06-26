"""
TileLang batched vector-matrix product.
A: [M, K], B: [M, N, K]
Output[m, n] = sum_k(A[m, k] * B[m, n, k])
Equivalent to torch.einsum('mk,mnk->mn', A, B)

Strategy: Reshape B from [M, N, K] to [M*N, K] and use elementwise
multiply + reduce. Tile over M*N with reduction over K.

We avoid T.gemm to keep full float32 precision (T.gemm uses TF32).
"""
import torch
import tilelang
import tilelang.language as T

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("batched_matmul", "tilelang") or {}
except Exception:
    _TUNED = {}


_TL_DTYPE = {torch.float16: "float16", torch.float32: "float32", torch.bfloat16: "bfloat16"}


@tilelang.jit
def _batched_kernel(M_dim, N_dim, K_dim, bN, bK, threads, dtype):
    """C[m,n] = sum_k A[m,k] * B[m,n,k].

    One block per (m, n-tile of bN). Load A[m,:] into shared once (reused
    across the tile), stream B in (bN,bK) tiles via T.copy, multiply by the
    matching A slice, and T.reduce over K. Reads B in its native dtype (the old
    kernel upcast B to fp32 -> 2x the bytes) and accumulates in fp32. The op is
    bandwidth-bound on B, so this matches/beats torch.einsum.
    """
    @T.prim_func
    def func(
        A_t: T.Tensor((M_dim, K_dim), dtype),
        B_t: T.Tensor((M_dim, N_dim, K_dim), dtype),
        C_t: T.Tensor((M_dim, N_dim), dtype),
    ):
        with T.Kernel(T.ceildiv(N_dim, bN), M_dim, threads=threads) as (bx, by):
            A_sh = T.alloc_shared((K_dim,), dtype)
            Bt = T.alloc_shared((bN, bK), dtype)
            prod = T.alloc_fragment((bN, bK), "float32")
            part = T.alloc_fragment((bN,), "float32")
            acc = T.alloc_fragment((bN,), "float32")

            for k in T.Parallel(K_dim):
                A_sh[k] = A_t[by, k]
            T.clear(acc)
            for kt in T.serial(K_dim // bK):
                T.copy(B_t[by, bx * bN, kt * bK], Bt)
                for i, j in T.Parallel(bN, bK):
                    prod[i, j] = T.cast(Bt[i, j], "float32") * T.cast(A_sh[kt * bK + j], "float32")
                T.reduce(prod, part, "sum", dim=1, clear=True)
                for i in T.Parallel(bN):
                    acc[i] = acc[i] + part[i]
            for i in T.Parallel(bN):
                C_t[by, bx * bN + i] = T.cast(acc[i], dtype)

    return func


def batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.ndim == 2 and B.ndim == 3
    orig_dtype = A.dtype
    work_dtype = orig_dtype if orig_dtype in _TL_DTYPE else torch.float32
    A = A.to(work_dtype)
    B = B.to(work_dtype)
    M, K = A.shape
    M2, N, K2 = B.shape
    assert M == M2 and K == K2

    block_N = 64
    block_K = 256

    N_pad = ((N + block_N - 1) // block_N) * block_N
    K_pad = ((K + block_K - 1) // block_K) * block_K

    if N_pad != N or K_pad != K:
        B_pad = torch.zeros(M, N_pad, K_pad, device=B.device, dtype=work_dtype)
        B_pad[:, :N, :K] = B
    else:
        B_pad = B.contiguous()

    if K_pad != K:
        A_pad = torch.zeros(M, K_pad, device=A.device, dtype=work_dtype)
        A_pad[:, :K] = A
    else:
        A_pad = A.contiguous()

    C_pad = torch.empty(M, N_pad, device=A.device, dtype=work_dtype)

    fn = _batched_kernel(M, N_pad, K_pad, block_N, block_K, 128, _TL_DTYPE[work_dtype])
    fn(A_pad, B_pad, C_pad)

    return C_pad[:, :N].to(orig_dtype)
