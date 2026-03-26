"""
kernels/gemm/tilelang_impl.py
TileLang GEMM: C = A @ B  (fp16 in, fp32 accumulate, fp16 out)
"""
import torch
import tilelang
import tilelang.language as T

# ── Tile sizes (tune for your GPU) ────────────────────────────────────────────
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_STAGES = 3
NUM_THREADS = 128


@tilelang.jit
def _build(M, N, K, block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
           dtype="float16", accum_dtype="float"):

    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=NUM_THREADS
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=NUM_STAGES):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


def run(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dim mismatch: {K} vs {K2}"
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)
    kernel = _build(M, N, K)
    kernel(A, B, C)
    return C
