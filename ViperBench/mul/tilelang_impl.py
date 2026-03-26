import tilelang
import tilelang.language as T
import torch


def mul(x: torch.Tensor) -> torch.Tensor:
    """Multiply input tensor by 2 using TileLang."""
    N = x.numel()
    if N == 0:
        return torch.zeros_like(x)

    original_shape = x.shape
    block_N = 1024
    dtype_str = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(x.dtype, "float32")

    @tilelang.jit
    def kernel(n, block_size=block_N):
        @T.prim_func
        def func(
            A: T.Tensor((n,), dtype_str),
            C: T.Tensor((n,), dtype_str),
        ):
            with T.Kernel(T.ceildiv(n, block_size), threads=128) as bx:
                A_local = T.alloc_fragment((block_size,), dtype_str)
                T.copy(A[bx * block_size], A_local)
                for i in T.Parallel(block_size):
                    A_local[i] = A_local[i] * 2
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

    return c_pad[:N].view(original_shape)
