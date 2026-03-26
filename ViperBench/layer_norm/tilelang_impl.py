import tilelang
import tilelang.language as T
import torch

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("layer_norm", "tilelang") or {}
except Exception:
    _TUNED = {}


@tilelang.jit
def _layer_norm_kernel(m, n, threads=_TUNED.get("threads", 128)):
    eps_val = 1e-5

    @T.prim_func
    def func(
        X: T.Tensor((m, n), "float32"),
        W: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
        Y: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(m, threads=threads) as bx:
            row = T.alloc_fragment((n,), "float32")
            mean_val = T.alloc_fragment((1,), "float32")
            var_val = T.alloc_fragment((1,), "float32")
            w_frag = T.alloc_fragment((n,), "float32")
            b_frag = T.alloc_fragment((n,), "float32")

            # Load weight/bias
            for j in T.Parallel(n):
                w_frag[j] = W[j]
            for j in T.Parallel(n):
                b_frag[j] = B[j]

            # Load row
            for j in T.Parallel(n):
                row[j] = X[bx, j]

            # Compute mean
            mean_val[0] = T.float32(0)
            for j in T.serial(n):
                mean_val[0] = mean_val[0] + row[j]
            mean_val[0] = mean_val[0] / T.float32(n)

            # Compute variance
            var_val[0] = T.float32(0)
            for j in T.serial(n):
                var_val[0] = var_val[0] + (row[j] - mean_val[0]) * (row[j] - mean_val[0])
            var_val[0] = var_val[0] / T.float32(n)

            # Normalize: (x - mean) / sqrt(var + eps) * weight + bias
            for j in T.Parallel(n):
                row[j] = (row[j] - mean_val[0]) / T.sqrt(var_val[0] + T.float32(eps_val)) * w_frag[j] + b_frag[j]

            # Store row
            for j in T.Parallel(n):
                Y[bx, j] = row[j]
    return func


def layer_norm(x, weight, bias, eps=1e-5):
    """TileLang layer norm: y = (x - mean) / sqrt(var + eps) * weight + bias
    Unified API: layer_norm(x, weight, bias, eps) -> Tensor
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    M = x.numel() // D

    x_2d = x.contiguous().float().reshape(M, D)
    w = weight.float().contiguous()
    b = bias.float().contiguous()

    y = torch.zeros(M, D, device=x.device, dtype=torch.float32)
    func = _layer_norm_kernel(M, D)
    func(x_2d, w, b, y)

    return y.reshape(orig_shape).to(x.dtype)
