import tilelang
import tilelang.language as T
import torch
import math

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("max_reduction", "tilelang") or {}
except Exception:
    _TUNED = {}


def max_reduction(input, dim, keepdim=False):
    """Max reduction along dim, returns (values, indices) like torch.max."""
    if not isinstance(input, torch.Tensor):
        raise TypeError('The input must be a torch.Tensor.')

    x = input
    dim = dim % x.ndim
    shape = x.shape
    N = shape[dim]
    M = math.prod(shape[:dim]) if dim > 0 else 1
    K = x.numel() // M // N

    x_3d = x.contiguous().float().view(M, N, K)

    block_M = min(_TUNED.get("block_M", 32), M)
    block_K = min(_TUNED.get("block_K", 32), K)
    M_pad = ((M + block_M - 1) // block_M) * block_M
    K_pad = ((K + block_K - 1) // block_K) * block_K

    if M_pad != M or K_pad != K:
        x_pad = torch.full((M_pad, N, K_pad), float('-inf'), device=x.device, dtype=torch.float32)
        x_pad[:M, :, :K] = x_3d
        val_pad = torch.full((M_pad, K_pad), float('-inf'), device=x.device, dtype=torch.float32)
        idx_pad = torch.zeros(M_pad, K_pad, dtype=torch.int32, device=x.device)
    else:
        x_pad = x_3d
        val_pad = torch.full((M, K), float('-inf'), device=x.device, dtype=torch.float32)
        idx_pad = torch.zeros(M, K, dtype=torch.int32, device=x.device)

    @tilelang.jit
    def kernel(m, n, k, bM=block_M, bK=block_K):
        @T.prim_func
        def func(
            A: T.Tensor((m, n, k), "float32"),
            Val: T.Tensor((m, k), "float32"),
            Idx: T.Tensor((m, k), "int32"),
        ):
            with T.Kernel(T.ceildiv(k, bK), T.ceildiv(m, bM), threads=_TUNED.get("threads", 128)) as (bx, by):
                max_val = T.alloc_fragment((bM, bK), "float32")
                max_idx = T.alloc_fragment((bM, bK), "int32")
                cur_val = T.alloc_fragment((bM, bK), "float32")
                T.clear(max_idx)
                for i, j in T.Parallel(bM, bK):
                    max_val[i, j] = T.float32(-1e30)
                for ni in T.serial(n):
                    for i, j in T.Parallel(bM, bK):
                        cur_val[i, j] = A[by * bM + i, ni, bx * bK + j]
                    for i, j in T.Parallel(bM, bK):
                        if cur_val[i, j] > max_val[i, j]:
                            max_val[i, j] = cur_val[i, j]
                            max_idx[i, j] = T.int32(ni)
                T.copy(max_val, Val[by * bM, bx * bK])
                T.copy(max_idx, Idx[by * bM, bx * bK])
        return func

    func = kernel(M_pad, N, K_pad)
    func(x_pad, val_pad, idx_pad)

    vals = val_pad[:M, :K].to(x.dtype)
    idxs = idx_pad[:M, :K].to(torch.int64)

    shape_list = list(shape)
    shape_list[dim] = 1
    vals = vals.view(shape_list)
    idxs = idxs.view(shape_list)
    if not keepdim:
        vals = vals.squeeze(dim)
        idxs = idxs.squeeze(dim)
    return (vals, idxs)
