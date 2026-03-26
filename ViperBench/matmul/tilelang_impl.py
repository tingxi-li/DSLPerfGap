import tilelang
import tilelang.language as T
import torch

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("matmul", "tilelang") or {}
except Exception:
    _TUNED = {}

_threads = _TUNED.get("threads", 128)
_num_stages = _TUNED.get("num_stages", 3)


@tilelang.jit
def _gemm(M, N, K, block_M=_TUNED.get("block_M", 128), block_N=_TUNED.get("block_N", 128), block_K=_TUNED.get("block_K", 32)):
    @T.prim_func
    def func(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=_num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return func


def matmul(a, b):
    """Unified API: matmul(a, b) -> Tensor for 2D float16 matrices."""
    assert a.ndim == 2 and b.ndim == 2, "Only 2D matrices supported"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    block_M, block_N, block_K = _TUNED.get("block_M", 128), _TUNED.get("block_N", 128), _TUNED.get("block_K", 32)
    M_pad = ((M + block_M - 1) // block_M) * block_M
    N_pad = ((N + block_N - 1) // block_N) * block_N
    K_pad = ((K + block_K - 1) // block_K) * block_K

    a_c = a.half().contiguous()
    b_c = b.half().contiguous()

    if M_pad != M or K_pad != K:
        a_pad = torch.zeros(M_pad, K_pad, device=a.device, dtype=torch.float16)
        a_pad[:M, :K] = a_c
    else:
        a_pad = a_c

    if K_pad != K or N_pad != N:
        b_pad = torch.zeros(K_pad, N_pad, device=b.device, dtype=torch.float16)
        b_pad[:K, :N] = b_c
    else:
        b_pad = b_c

    c_pad = torch.zeros(M_pad, N_pad, device=a.device, dtype=torch.float16)
    func = _gemm(M_pad, N_pad, K_pad)
    func(a_pad, b_pad, c_pad)
    return c_pad[:M, :N]
