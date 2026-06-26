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


@tilelang.jit
def _max_reduction_kernel(m, n, k, bM=_TUNED.get("block_M", 32), bK=_TUNED.get("block_K", 32), threads=_TUNED.get("threads", 128)):
    @T.prim_func
    def func(
        A: T.Tensor((m, n, k), "float32"),
        Val: T.Tensor((m, k), "float32"),
        Idx: T.Tensor((m, k), "int32"),
    ):
        with T.Kernel(T.ceildiv(k, bK), T.ceildiv(m, bM), threads=threads) as (bx, by):
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


_TL_DTYPE = {torch.float16: "float16", torch.float32: "float32", torch.bfloat16: "bfloat16"}


@tilelang.jit
def _max_val_kernel(M, N, blk, threads, dtype):
    """Pass 1: per-row max value (reduce over contiguous last dim, K==1).

    Reads input in its native dtype (no fp32 upcast => half the bytes for
    fp16/bf16), reduces in fp32 via tiled T.reduce. ~2x faster than torch.max
    because it skips the index.
    """
    nt = N // blk

    @T.prim_func
    def func(A: T.Tensor((M, N), dtype), Mx: T.Tensor((M,), "float32")):
        with T.Kernel(M, threads=threads) as bx:
            tile = T.alloc_fragment((blk,), "float32")
            part = T.alloc_fragment((1,), "float32")
            acc = T.alloc_fragment((1,), "float32")
            acc[0] = T.float32(-3.0e38)
            for t in T.serial(nt):
                for j in T.Parallel(blk):
                    tile[j] = T.cast(A[bx, t * blk + j], "float32")
                T.reduce(tile, part, "max", dim=0, clear=True)
                acc[0] = T.max(acc[0], part[0])
            Mx[bx] = acc[0]
    return func


@tilelang.jit
def _first_idx_kernel(M, N, blk, threads, dtype):
    """Pass 2: smallest column index whose value equals the row max.

    Matches torch.max tie-breaking (first occurrence) via reduce-min over
    candidate indices.
    """
    nt = N // blk

    @T.prim_func
    def func(A: T.Tensor((M, N), dtype), Mx: T.Tensor((M,), "float32"), Idx: T.Tensor((M,), "int32")):
        with T.Kernel(M, threads=threads) as bx:
            cand = T.alloc_fragment((blk,), "int32")
            part = T.alloc_fragment((1,), "int32")
            best = T.alloc_fragment((1,), "int32")
            mxv = T.alloc_fragment((1,), "float32")
            best[0] = T.int32(N)
            mxv[0] = Mx[bx]
            for t in T.serial(nt):
                for j in T.Parallel(blk):
                    if T.cast(A[bx, t * blk + j], "float32") >= mxv[0]:
                        cand[j] = T.int32(t * blk + j)
                    else:
                        cand[j] = T.int32(N)
                T.reduce(cand, part, "min", dim=0, clear=True)
                best[0] = T.min(best[0], part[0])
            Idx[bx] = best[0]
    return func


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

    # Fast path: reduction over the contiguous last dim (K == 1). Two
    # bandwidth-bound passes (max value, then first index) instead of a serial
    # element scan. Reads native dtype; reaches torch.max parity on large rows.
    if K == 1 and x.dtype in _TL_DTYPE:
        blk = min(4096, N)
        if blk > 0 and N % blk == 0 and (blk & (blk - 1)) == 0:
            tl_dtype = _TL_DTYPE[x.dtype]
            x_2d = x.contiguous().view(M, N)
            mx = torch.empty(M, device=x.device, dtype=torch.float32)
            idx = torch.empty(M, device=x.device, dtype=torch.int32)
            _max_val_kernel(M, N, blk, 256, tl_dtype)(x_2d, mx)
            _first_idx_kernel(M, N, blk, 256, tl_dtype)(x_2d, mx, idx)
            vals = mx.to(x.dtype).view(M, 1)
            idxs = idx.to(torch.int64).view(M, 1)
            shape_list = list(shape)
            shape_list[dim] = 1
            vals = vals.view(shape_list)
            idxs = idxs.view(shape_list)
            if not keepdim:
                vals = vals.squeeze(dim)
                idxs = idxs.squeeze(dim)
            return (vals, idxs)

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

    func = _max_reduction_kernel(M_pad, N, K_pad, bM=block_M, bK=block_K)
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
