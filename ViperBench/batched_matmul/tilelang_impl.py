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


def batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.ndim == 2 and B.ndim == 3
    orig_dtype = A.dtype
    A = A.float()
    B = B.float()
    M, K = A.shape
    M2, N, K2 = B.shape
    assert M == M2 and K == K2

    # Flatten to 2D problem:
    # A_flat[m*N+n, k] = A[m, k]  (broadcast A across N)
    # B_flat[m*N+n, k] = B[m, n, k]
    # C_flat[m*N+n] = sum_k A_flat[m*N+n, k] * B_flat[m*N+n, k]
    #
    # But this requires expanding A which wastes memory.
    # Instead, use tiled approach: grid over (M, N_blocks), reduce over K.

    block_N = _TUNED.get("block_N", 64)
    block_K = _TUNED.get("block_K", 64)

    N_pad = ((N + block_N - 1) // block_N) * block_N
    K_pad = ((K + block_K - 1) // block_K) * block_K

    # Pad B to [M, N_pad, K_pad]
    if N_pad != N or K_pad != K:
        B_pad = torch.zeros(M, N_pad, K_pad, device=B.device, dtype=B.dtype)
        B_pad[:, :N, :K] = B
    else:
        B_pad = B.contiguous()

    # Pad A to [M, K_pad]
    if K_pad != K:
        A_pad = torch.zeros(M, K_pad, device=A.device, dtype=A.dtype)
        A_pad[:, :K] = A
    else:
        A_pad = A.contiguous()

    C_pad = torch.zeros(M, N_pad, device=A.device, dtype=A.dtype)

    @tilelang.jit
    def kernel(M_dim, N_dim, K_dim, bN=block_N, bK=block_K):
        @T.prim_func
        def func(
            A_t: T.Tensor((M_dim, K_dim), "float32"),
            B_t: T.Tensor((M_dim, N_dim, K_dim), "float32"),
            C_t: T.Tensor((M_dim, N_dim), "float32"),
        ):
            # Grid: (N_blocks, M) - one thread block per (m, n_block)
            with T.Kernel(
                T.ceildiv(N_dim, bN), M_dim,
                threads=_TUNED.get("threads", 128)
            ) as (bx, by):
                C_local = T.alloc_fragment((bN,), "float32")
                A_local = T.alloc_fragment((bK,), "float32")
                B_local = T.alloc_fragment((bN,), "float32")

                T.clear(C_local)

                for k_tile in range(T.ceildiv(K_dim, bK)):
                    # Load A tile for this m
                    for j in T.Parallel(bK):
                        A_local[j] = A_t[by, k_tile * bK + j]

                    # For each k in the tile, multiply and accumulate
                    for kk in range(bK):
                        for j in T.Parallel(bN):
                            B_local[j] = B_t[by, bx * bN + j, k_tile * bK + kk]
                        for j in T.Parallel(bN):
                            C_local[j] += A_local[kk] * B_local[j]

                # Store result
                for j in T.Parallel(bN):
                    C_t[by, bx * bN + j] = C_local[j]

        return func

    fn = kernel(M, N_pad, K_pad)
    fn(A_pad, B_pad, C_pad)

    return C_pad[:, :N].to(orig_dtype)
