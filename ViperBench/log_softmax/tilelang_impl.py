import tilelang
import tilelang.language as T
import torch

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("log_softmax", "tilelang") or {}
except Exception:
    _TUNED = {}


@tilelang.jit
def _log_softmax_kernel(m, n, threads=_TUNED.get("threads", 128)):
    @T.prim_func
    def func(A: T.Tensor((m, n), "float32"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(m, threads=threads) as bx:
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


_TL_DTYPE = {torch.float16: "float16", torch.float32: "float32", torch.bfloat16: "bfloat16"}


@tilelang.jit
def _log_softmax_row_kernel(M, N, blk, threads, dtype):
    """One block per row; row cached in shared memory, tiled reductions.

    Reads native dtype directly (no fp32 host-side copy) and does one global
    read + one global write. out = x - max - log(sum exp(x - max)).
    """
    nt = N // blk

    @T.prim_func
    def func(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(M, threads=threads) as bx:
            S = T.alloc_shared((N,), dtype)
            tile = T.alloc_fragment((blk,), "float32")
            part = T.alloc_fragment((1,), "float32")
            mx = T.alloc_fragment((1,), "float32")
            sm = T.alloc_fragment((1,), "float32")
            lse = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(N):
                S[j] = A[bx, j]
            mx[0] = T.float32(-3.0e38)
            for t in T.serial(nt):
                for j in T.Parallel(blk):
                    tile[j] = T.cast(S[t * blk + j], "float32")
                T.reduce(tile, part, "max", dim=0, clear=True)
                mx[0] = T.max(mx[0], part[0])
            sm[0] = T.float32(0)
            for t in T.serial(nt):
                for j in T.Parallel(blk):
                    tile[j] = T.exp(T.cast(S[t * blk + j], "float32") - mx[0])
                T.reduce(tile, part, "sum", dim=0, clear=True)
                sm[0] = sm[0] + part[0]
            lse[0] = T.log(sm[0]) + mx[0]
            for t in T.serial(nt):
                for j in T.Parallel(blk):
                    C[bx, t * blk + j] = T.cast(T.cast(S[t * blk + j], "float32") - lse[0], dtype)
    return func


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
    out_dtype = dtype if dtype is not None else x.dtype

    # Fast path: native-dtype shared-memory row kernel (no fp32 round-trip).
    if x.dtype in _TL_DTYPE and N > 0:
        blk = min(4096, N)
        # require a power-of-2 tile so the (blk,) -> threads reduce layout is valid
        if N % blk == 0 and (blk & (blk - 1)) == 0:
            out = torch.empty(M, N, device=x.device, dtype=x.dtype)
            _log_softmax_row_kernel(M, N, blk, 256, _TL_DTYPE[x.dtype])(x_2d, out)
            result = out.view(outer, trailing, inner).permute(0, 2, 1).contiguous().view(x.shape)
            return result.to(out_dtype)

    x_f32 = x_2d.float().contiguous()
    c_out = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    func = _log_softmax_kernel(M, N)
    func(x_f32, c_out)

    result = c_out.view(outer, trailing, inner).permute(0, 2, 1).contiguous().view(x.shape)
    return result.to(out_dtype)
