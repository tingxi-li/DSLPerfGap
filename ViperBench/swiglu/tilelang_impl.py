import tilelang
import tilelang.language as T
import torch

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("swiglu", "tilelang") or {}
except Exception:
    _TUNED = {}


def swiglu(xy, out=None):
    """
    SwiGLU forward pass using TileLang.
    Splits input along last dim into x, y; computes x * sigmoid(x) * y.
    """
    assert xy.shape[-1] % 2 == 0
    orig_shape = xy.shape
    D = xy.shape[-1] // 2
    M = xy.numel() // xy.shape[-1]

    # Flatten to 2D and split
    xy_2d = xy.contiguous().float().view(M, 2 * D)
    x_part = xy_2d[:, :D].contiguous()
    y_part = xy_2d[:, D:].contiguous()

    block_M = min(_TUNED.get("block_M", 32), M)
    M_pad = ((M + block_M - 1) // block_M) * block_M

    @tilelang.jit
    def kernel(m, d, bM=block_M):
        @T.prim_func
        def func(
            X: T.Tensor((m, d), "float32"),
            Y: T.Tensor((m, d), "float32"),
            Out: T.Tensor((m, d), "float32"),
        ):
            with T.Kernel(T.ceildiv(m, bM), threads=_TUNED.get("threads", 128)) as bx:
                x_frag = T.alloc_fragment((bM, d), "float32")
                y_frag = T.alloc_fragment((bM, d), "float32")
                T.copy(X[bx * bM, 0], x_frag)
                T.copy(Y[bx * bM, 0], y_frag)
                for i, j in T.Parallel(bM, d):
                    x_val = x_frag[i, j]
                    sig = T.float32(1.0) / (T.float32(1.0) + T.exp(-x_val))
                    x_frag[i, j] = x_val * sig * y_frag[i, j]
                T.copy(x_frag, Out[bx * bM, 0])
        return func

    if M_pad != M:
        x_pad = torch.zeros(M_pad, D, device=xy.device, dtype=torch.float32)
        y_pad = torch.zeros(M_pad, D, device=xy.device, dtype=torch.float32)
        x_pad[:M, :] = x_part
        y_pad[:M, :] = y_part
    else:
        x_pad = x_part
        y_pad = y_part

    out_pad = torch.zeros(M_pad, D, device=xy.device, dtype=torch.float32)
    func = kernel(M_pad, D)
    func(x_pad, y_pad, out_pad)

    out_shape = list(orig_shape)
    out_shape[-1] = D
    result = out_pad[:M, :].view(out_shape).to(xy.dtype)

    if out is not None:
        out.copy_(result)
        return out
    return result
