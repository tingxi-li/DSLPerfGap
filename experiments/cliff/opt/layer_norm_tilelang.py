import torch
import torch.nn as nn

# --- Original implementation inlined below ---
import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-1])
def _layer_norm_kernel(m, n, threads=256):
    eps_val = 1e-5

    @T.prim_func
    def func(
        X: T.Tensor((m, n), "bfloat16"),
        W: T.Tensor((n,), "bfloat16"),
        B: T.Tensor((n,), "bfloat16"),
        Y: T.Tensor((m, n), "bfloat16"),
    ):
        with T.Kernel(m, threads=threads) as bx:
            row = T.alloc_fragment((n,), "float32")
            mean_val = T.alloc_fragment((1,), "float32")
            var_val = T.alloc_fragment((1,), "float32")
            out_frag = T.alloc_fragment((n,), "bfloat16")

            # Load row (cast to float32)
            for j in T.Parallel(n):
                row[j] = T.cast(X[bx, j], "float32")

            # Compute mean
            T.reduce(row, mean_val, "sum", dim=0, clear=True)
            mean_val[0] = mean_val[0] / T.float32(n)

            # Subtract mean in-place
            for j in T.Parallel(n):
                row[j] = row[j] - mean_val[0]

            # Save (x-mean) as bfloat16
            for j in T.Parallel(n):
                out_frag[j] = T.cast(row[j], "bfloat16")

            # Square for variance
            for j in T.Parallel(n):
                row[j] = row[j] * row[j]

            T.reduce(row, var_val, "sum", dim=0, clear=True)
            var_val[0] = var_val[0] / T.float32(n)

            # Normalize
            for j in T.Parallel(n):
                out_frag[j] = T.cast(
                    T.cast(out_frag[j], "float32") / T.sqrt(var_val[0] + T.float32(eps_val)) * T.cast(W[j], "float32") + T.cast(B[j], "float32"),
                    "bfloat16"
                )

            T.copy(out_frag, Y[bx, 0:n])
    return func


def layer_norm(x, weight, bias, eps=1e-5):
    """TileLang layer norm: y = (x - mean) / sqrt(var + eps) * weight + bias"""
    orig_shape = x.shape
    D = orig_shape[-1]
    M = x.numel() // D

    x_2d = x.contiguous().reshape(M, D)
    w = weight.contiguous()
    b = bias.contiguous()

    func = _layer_norm_kernel(M, D)
    y = func(x_2d, w, b)

    return y.reshape(orig_shape)

# --- End original implementation ---


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return layer_norm(*args)


def get_inputs():
    x = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(8192, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(8192, device='cuda', dtype=torch.bfloat16)
    return [x, weight, bias]

def get_init_inputs():
    return []
