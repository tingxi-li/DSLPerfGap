import tilelang
import tilelang.language as T
import torch


def matrix_transpose(x):
    """Unified API: matrix_transpose(x) -> Tensor, returns x.T contiguous.
    Only supports 2D input matrices.
    """
    if x.ndim != 2:
        raise ValueError(f"Only 2D matrices supported, got {x.ndim}D tensor with shape {x.shape}")

    M, N = x.shape
    dtype_str = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(x.dtype, "float32")

    block_M = 64
    block_N = 64

    M_pad = ((M + block_M - 1) // block_M) * block_M
    N_pad = ((N + block_N - 1) // block_N) * block_N

    @tilelang.jit
    def kernel(m, n, bM=block_M, bN=block_N):
        @T.prim_func
        def func(A: T.Tensor((m, n), dtype_str), B: T.Tensor((n, m), dtype_str)):
            with T.Kernel(T.ceildiv(n, bN), T.ceildiv(m, bM), threads=128) as (bx, by):
                A_local = T.alloc_fragment((bM, bN), dtype_str)
                B_local = T.alloc_fragment((bN, bM), dtype_str)
                T.copy(A[by * bM, bx * bN], A_local)
                for i, j in T.Parallel(bN, bM):
                    B_local[i, j] = A_local[j, i]
                T.copy(B_local, B[bx * bN, by * bM])
        return func

    if M_pad != M or N_pad != N:
        x_pad = torch.zeros(M_pad, N_pad, device=x.device, dtype=x.dtype)
        x_pad[:M, :N] = x
    else:
        x_pad = x.contiguous()

    out_pad = torch.zeros(N_pad, M_pad, device=x.device, dtype=x.dtype)
    func = kernel(M_pad, N_pad)
    func(x_pad, out_pad)
    return out_pad[:N, :M].contiguous()
