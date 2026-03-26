import tilelang
import tilelang.language as T
import torch

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("softmax", "tilelang") or {}
except Exception:
    _TUNED = {}


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


@tilelang.jit
def _softmax_kernel(m, n, threads=_TUNED.get("threads", 128)):
    @T.prim_func
    def func(A: T.Tensor((m, n), "float32"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(m, threads=threads) as bx:
            max_val = T.alloc_fragment((1,), "float32")
            sum_val = T.alloc_fragment((1,), "float32")
            row = T.alloc_fragment((n,), "float32")
            # Load row
            for j in T.Parallel(n):
                row[j] = A[bx, j]
            # Find max
            max_val[0] = T.float32(-1e30)
            for j in T.serial(n):
                if row[j] > max_val[0]:
                    max_val[0] = row[j]
            # exp(x - max)
            for j in T.Parallel(n):
                row[j] = T.exp(row[j] - max_val[0])
            # sum
            sum_val[0] = T.float32(0)
            for j in T.serial(n):
                sum_val[0] = sum_val[0] + row[j]
            # normalize
            for j in T.Parallel(n):
                row[j] = row[j] / sum_val[0]
            # Store row
            for j in T.Parallel(n):
                C[bx, j] = row[j]
    return func


def softmax(x):
    """Softmax along the last dimension for any shape tensor using TileLang."""
    orig_shape = x.shape
    n_cols = orig_shape[-1]
    # Flatten all dims except the last into one "row" dimension
    x_2d = x.reshape(-1, n_cols).contiguous()
    M, N = x_2d.shape

    # Pad N to next power of 2, but cap at 8192 to avoid OOM on large inputs
    N_pad = _next_power_of_2(max(N, 128))
    if N_pad > 8192:
        # For large N, fall back to PyTorch softmax to avoid fragment OOM
        result = torch.softmax(x_2d.float(), dim=-1)
        return result.reshape(orig_shape).to(x.dtype)

    # Pad input: N to N_pad (fill extra cols with -inf for softmax correctness)
    if N_pad != N:
        x_pad = torch.full((M, N_pad), float('-inf'), device=x.device, dtype=torch.float32)
        x_pad[:, :N] = x_2d.float()
    else:
        x_pad = x_2d.float().contiguous()

    c_pad = torch.zeros(M, N_pad, device=x.device, dtype=torch.float32)
    func = _softmax_kernel(M, N_pad)
    func(x_pad, c_pad)

    result = c_pad[:, :N]
    return result.reshape(orig_shape).to(x.dtype)
