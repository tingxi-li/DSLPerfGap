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


@tilelang.jit
def _swiglu_kernel_f16(m, n, bM=32, bN=256, threads=128):
    @T.prim_func
    def func(
        X: T.Tensor((m, n), "float16"),
        Y: T.Tensor((m, n), "float16"),
        Out: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(T.ceildiv(n, bN), T.ceildiv(m, bM), threads=threads) as (bx, by):
            x_frag = T.alloc_fragment((bM, bN), "float32")
            y_frag = T.alloc_fragment((bM, bN), "float32")
            T.copy(X[by * bM, bx * bN], x_frag)
            T.copy(Y[by * bM, bx * bN], y_frag)
            for i, j in T.Parallel(bM, bN):
                x_val = x_frag[i, j]
                sig = T.float32(1.0) / (T.float32(1.0) + T.exp(-x_val))
                x_frag[i, j] = x_val * sig * y_frag[i, j]
            o_frag = T.alloc_fragment((bM, bN), "float16")
            for i, j in T.Parallel(bM, bN):
                o_frag[i, j] = T.cast(x_frag[i, j], "float16")
            T.copy(o_frag, Out[by * bM, bx * bN])
    return func


@tilelang.jit
def _swiglu_kernel_f32(m, n, bM=32, bN=256, threads=128):
    @T.prim_func
    def func(
        X: T.Tensor((m, n), "float32"),
        Y: T.Tensor((m, n), "float32"),
        Out: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(T.ceildiv(n, bN), T.ceildiv(m, bM), threads=threads) as (bx, by):
            x_frag = T.alloc_fragment((bM, bN), "float32")
            y_frag = T.alloc_fragment((bM, bN), "float32")
            T.copy(X[by * bM, bx * bN], x_frag)
            T.copy(Y[by * bM, bx * bN], y_frag)
            for i, j in T.Parallel(bM, bN):
                x_val = x_frag[i, j]
                sig = T.float32(1.0) / (T.float32(1.0) + T.exp(-x_val))
                x_frag[i, j] = x_val * sig * y_frag[i, j]
            T.copy(x_frag, Out[by * bM, bx * bN])
    return func


def swiglu(xy, out=None):
    """
    SwiGLU forward pass using TileLang.
    Splits input along last dim into x, y; computes x * sigmoid(x) * y.
    """
    assert xy.shape[-1] % 2 == 0
    orig_shape = xy.shape
    D = xy.shape[-1] // 2
    M = xy.numel() // xy.shape[-1]
    input_dtype = xy.dtype

    # Flatten to 2D and split
    xy_2d = xy.contiguous().view(M, 2 * D)
    x_part = xy_2d[:, :D].contiguous()
    y_part = xy_2d[:, D:].contiguous()

    block_M = min(_TUNED.get("block_M", 32), M)
    block_N = min(_TUNED.get("block_N", 256), D)
    threads = _TUNED.get("threads", 128)

    M_pad = ((M + block_M - 1) // block_M) * block_M
    D_pad = ((D + block_N - 1) // block_N) * block_N

    needs_m_pad = M_pad != M
    needs_d_pad = D_pad != D

    if input_dtype == torch.float16:
        pad_dtype = torch.float16
        kernel_fn = _swiglu_kernel_f16
    else:
        # For float32 and bfloat16, use float32 path
        pad_dtype = torch.float32
        kernel_fn = _swiglu_kernel_f32
        if input_dtype != torch.float32:
            x_part = x_part.float()
            y_part = y_part.float()

    if needs_m_pad or needs_d_pad:
        x_pad = torch.zeros(M_pad, D_pad, device=xy.device, dtype=pad_dtype)
        y_pad = torch.zeros(M_pad, D_pad, device=xy.device, dtype=pad_dtype)
        x_pad[:M, :D] = x_part
        y_pad[:M, :D] = y_part
    else:
        x_pad = x_part
        y_pad = y_part

    out_pad = torch.zeros(M_pad, D_pad, device=xy.device, dtype=pad_dtype)
    func = kernel_fn(M_pad, D_pad, bM=block_M, bN=block_N, threads=threads)
    func(x_pad, y_pad, out_pad)

    out_shape = list(orig_shape)
    out_shape[-1] = D
    result = out_pad[:M, :D].view(out_shape).to(input_dtype)

    if out is not None:
        out.copy_(result)
        return out
    return result
