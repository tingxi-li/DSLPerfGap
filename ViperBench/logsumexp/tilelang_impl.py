import tilelang
import tilelang.language as T
import torch

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("logsumexp", "tilelang") or {}
except Exception:
    _TUNED = {}


def logsumexp(x):
    """Logsumexp reduction along the last dimension using TileLang."""
    orig_shape = x.shape
    # Flatten to 2D: (M, N) where N is the last dimension
    N = orig_shape[-1]
    M = x.numel() // N
    x_2d = x.contiguous().view(M, N).float()

    # block_M must be >= 2 to avoid degenerate T.Parallel loops
    block_M = min(_TUNED.get("block_M", 32), max(2, M))
    M_pad = ((M + block_M - 1) // block_M) * block_M

    @tilelang.jit
    def kernel(m, n, bM=block_M):
        @T.prim_func
        def func(A: T.Tensor((m, n), "float32"), C: T.Tensor((m,), "float32")):
            with T.Kernel(T.ceildiv(m, bM), threads=_TUNED.get("threads", 128)) as bx:
                row = T.alloc_fragment((bM, n), "float32")
                max_val = T.alloc_fragment((bM,), "float32")
                sum_val = T.alloc_fragment((bM,), "float32")
                out_val = T.alloc_fragment((bM,), "float32")

                T.copy(A[bx * bM, 0], row)

                # Reduce max along dim=1 (the n dimension)
                T.reduce_max(row, max_val, dim=1)

                # Compute exp(row - max_val) in-place
                for i, j in T.Parallel(bM, n):
                    row[i, j] = T.exp(row[i, j] - max_val[i])

                # Reduce sum along dim=1
                T.reduce_sum(row, sum_val, dim=1)

                # Compute logsumexp = max + log(sum)
                for i in T.Parallel(bM):
                    out_val[i] = max_val[i] + T.log(sum_val[i])

                T.copy(out_val, C[bx * bM])
        return func

    if M_pad != M:
        x_pad = torch.full((M_pad, N), float('-inf'), device=x.device, dtype=torch.float32)
        x_pad[:M, :] = x_2d
    else:
        x_pad = x_2d.contiguous()

    c_pad = torch.zeros(M_pad, device=x.device, dtype=torch.float32)
    func = kernel(M_pad, N)
    func(x_pad, c_pad)

    result = c_pad[:M]
    out_shape = list(orig_shape[:-1])
    if len(out_shape) == 0:
        result = result.squeeze()
    else:
        result = result.view(*out_shape)
    return result
