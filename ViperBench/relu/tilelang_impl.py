import tilelang
import tilelang.language as T
import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    """Element-wise ReLU activation using TileLang."""
    N = x.numel()
    if N == 0:
        return torch.zeros_like(x)

    block_N = 1024
    dtype_str = "float32" if x.dtype == torch.float32 else "float16"

    @tilelang.jit
    def kernel(n, block_size=block_N):
        @T.prim_func
        def func(A: T.Tensor((n,), dtype_str), C: T.Tensor((n,), dtype_str)):
            with T.Kernel(T.ceildiv(n, block_size), threads=128) as bx:
                A_local = T.alloc_fragment((block_size,), dtype_str)
                T.copy(A[bx * block_size], A_local)
                for i in T.Parallel(block_size):
                    if A_local[i] < T.cast(0, dtype_str):
                        A_local[i] = T.cast(0, dtype_str)
                T.copy(A_local, C[bx * block_size])
        return func

    padded_N = ((N + block_N - 1) // block_N) * block_N
    x_flat = x.contiguous().view(-1)

    if padded_N != N:
        x_pad = torch.zeros(padded_N, device=x.device, dtype=x.dtype)
        x_pad[:N] = x_flat
    else:
        x_pad = x_flat

    c_pad = torch.zeros(padded_N, device=x.device, dtype=x.dtype)
    func = kernel(padded_N)
    func(x_pad, c_pad)

    return c_pad[:N].view(x.shape)
