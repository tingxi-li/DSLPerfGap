import torch
import torch.nn as nn

# --- Original implementation inlined below ---
import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def _rms_norm_kernel(m, n, threads=512):
    eps_val = 1e-5

    @T.prim_func
    def func(
        X: T.Tensor((m, n), "float16"),
        W: T.Tensor((n,), "float16"),
        Y: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(m, threads=threads) as bx:
            row = T.alloc_fragment((n,), "float32")
            sq_row = T.alloc_fragment((n,), "float32")
            rms_val = T.alloc_fragment((1,), "float32")

            # Load row into float32 for precision
            for j in T.Parallel(n):
                row[j] = T.cast(X[bx, j], "float32")

            # Compute squares
            for j in T.Parallel(n):
                sq_row[j] = row[j] * row[j]

            # Reduce sum of squares using T.reduce
            T.reduce(sq_row, rms_val, "sum", dim=0, clear=True)

            # Compute 1/RMS = rsqrt(mean(x^2) + eps)
            rms_val[0] = T.rsqrt(rms_val[0] / T.float32(n) + T.float32(eps_val))

            # Normalize and store: x * (1/rms) * weight, cast to fp16
            for j in T.Parallel(n):
                Y[bx, j] = T.cast(row[j] * rms_val[0] * T.cast(W[j], "float32"), "float16")
    return func


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    """RMS norm: y = x / sqrt(mean(x^2) + eps) * weight"""
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    N = 1
    for s in normalized_shape:
        N *= s
    M = x.numel() // N

    orig_shape = x.shape
    x_2d = x.contiguous().view(M, N).half()
    w = weight.half().contiguous()

    y = torch.empty(M, N, device=x.device, dtype=torch.float16)
    func = _rms_norm_kernel(M, N)
    func(x_2d, w, y)

    return y.view(orig_shape).to(x.dtype)

# --- End original implementation ---


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return rms_norm(*args)


def get_inputs():
    x = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
    normalized_shape = (8192,)
    weight = torch.randn(8192, device='cuda', dtype=torch.float16)
    return [x, normalized_shape, weight]

def get_init_inputs():
    return []
