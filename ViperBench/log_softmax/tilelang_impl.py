import tilelang
import tilelang.language as T
import torch


def log_softmax(x, dim=-1, dtype=None):
    """Log-softmax along the given dimension using TileLang."""
    dim = dim % x.ndim
    outer = 1
    for i in range(dim):
        outer *= x.shape[i]
    inner = x.shape[dim]
    trailing = 1
    for i in range(dim + 1, x.ndim):
        trailing *= x.shape[i]

    x_2d = x.contiguous().view(outer, inner, trailing).permute(0, 2, 1).contiguous().view(-1, inner)
    M, N = x_2d.shape

    @tilelang.jit
    def kernel(m, n):
        @T.prim_func
        def func(A: T.Tensor((m, n), "float32"), C: T.Tensor((m, n), "float32")):
            with T.Kernel(m, threads=128) as bx:
                row = T.alloc_fragment((n,), "float32")
                row_max = T.alloc_fragment((1,), "float32")
                row_sum = T.alloc_fragment((1,), "float32")
                T.copy(A[bx, 0:n], row)
                # Find max per row
                T.reduce(row, row_max, "max", dim=0, clear=True)
                # exp(x - max) and sum
                for i in T.Parallel(n):
                    row[i] = T.exp(row[i] - row_max[0])
                T.reduce(row, row_sum, "sum", dim=0, clear=True)
                # log_softmax = x - max - log(sum)
                T.copy(A[bx, 0:n], row)
                for i in T.Parallel(n):
                    row[i] = row[i] - row_max[0] - T.log(row_sum[0])
                T.copy(row, C[bx, 0:n])
        return func

    x_f32 = x_2d.float().contiguous()
    c_out = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    func = kernel(M, N)
    func(x_f32, c_out)

    result = c_out.view(outer, trailing, inner).permute(0, 2, 1).contiguous().view(x.shape)

    out_dtype = dtype if dtype is not None else x.dtype
    return result.to(out_dtype)
