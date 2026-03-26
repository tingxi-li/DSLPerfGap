import tilelang
import tilelang.language as T
import torch
import math


def _mean_single_dim(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Mean reduction along a single dim using TileLang."""
    dim = dim % x.ndim
    shape = x.shape
    N = shape[dim]
    M = math.prod(shape[:dim]) if dim > 0 else 1
    K = x.numel() // M // N

    x_3d = x.contiguous().float().view(M, N, K)

    block_M = min(32, M)
    block_K = min(32, K)
    M_pad = ((M + block_M - 1) // block_M) * block_M
    K_pad = ((K + block_K - 1) // block_K) * block_K

    if M_pad != M or K_pad != K:
        x_pad = torch.zeros(M_pad, N, K_pad, device=x.device, dtype=torch.float32)
        x_pad[:M, :, :K] = x_3d
        out_pad = torch.zeros(M_pad, K_pad, device=x.device, dtype=torch.float32)
    else:
        x_pad = x_3d
        out_pad = torch.zeros(M, K, device=x.device, dtype=torch.float32)

    @tilelang.jit
    def kernel(m, n, k, bM=block_M, bK=block_K):
        @T.prim_func
        def func(
            A: T.Tensor((m, n, k), "float32"),
            Out: T.Tensor((m, k), "float32"),
        ):
            with T.Kernel(T.ceildiv(k, bK), T.ceildiv(m, bM), threads=128) as (bx, by):
                sum_val = T.alloc_fragment((bM, bK), "float32")
                cur_val = T.alloc_fragment((bM, bK), "float32")
                T.clear(sum_val)
                for ni in T.serial(n):
                    for i, j in T.Parallel(bM, bK):
                        cur_val[i, j] = A[by * bM + i, ni, bx * bK + j]
                    for i, j in T.Parallel(bM, bK):
                        sum_val[i, j] = sum_val[i, j] + cur_val[i, j]
                for i, j in T.Parallel(bM, bK):
                    sum_val[i, j] = sum_val[i, j] / T.float32(n)
                T.copy(sum_val, Out[by * bM, bx * bK])
        return func

    func = kernel(M_pad, N, K_pad)
    func(x_pad, out_pad)

    result = out_pad[:M, :K]
    shape_list = list(shape)
    shape_list[dim] = 1
    result = result.view(shape_list)
    if not keepdim:
        result = result.squeeze(dim)
    return result


def mean_reduction(input_tensor, dim, keepdim=False, dtype=None):
    """
    Computes the mean value along specified dimensions using TileLang.
    Matches the signature of pytorch_impl.mean_reduction.
    """
    if dtype is None:
        out_dtype = input_tensor.dtype
    else:
        out_dtype = dtype

    if isinstance(dim, int):
        dims = [dim % input_tensor.ndim]
    else:
        dims = sorted([d % input_tensor.ndim for d in dim])

    # For multi-dim reduction, reduce dims one at a time from highest to lowest
    # so that earlier dim indices remain valid.
    result = input_tensor
    for d in reversed(sorted(dims)):
        result = _mean_single_dim(result, d, keepdim=True)

    if not keepdim:
        # Squeeze all reduced dims (squeeze from highest to lowest)
        for d in reversed(sorted(dims)):
            result = result.squeeze(d)

    return result.to(out_dtype)
