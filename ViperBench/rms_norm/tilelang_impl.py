import tilelang
import tilelang.language as T
import torch


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    """RMS norm: y = x / sqrt(mean(x^2) + eps) * weight"""
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    N = 1
    for s in normalized_shape:
        N *= s
    M = x.numel() // N

    orig_shape = x.shape
    x_2d = x.contiguous().float().view(M, N)
    w = weight.float().contiguous()

    @tilelang.jit
    def kernel(m, n):
        @T.prim_func
        def func(
            X: T.Tensor((m, n), "float32"),
            W: T.Tensor((n,), "float32"),
            Y: T.Tensor((m, n), "float32"),
        ):
            with T.Kernel(m, threads=128) as bx:
                row = T.alloc_fragment((n,), "float32")
                rms_val = T.alloc_fragment((1,), "float32")
                w_frag = T.alloc_fragment((n,), "float32")

                # Load weight
                for j in T.Parallel(n):
                    w_frag[j] = W[j]

                # Load row
                for j in T.Parallel(n):
                    row[j] = X[bx, j]

                # Compute mean of squares
                rms_val[0] = T.float32(0)
                for j in T.serial(n):
                    rms_val[0] = rms_val[0] + row[j] * row[j]
                rms_val[0] = T.sqrt(rms_val[0] / T.float32(n) + T.float32(eps))

                # Normalize: x / rms * weight
                for j in T.Parallel(n):
                    row[j] = row[j] / rms_val[0] * w_frag[j]

                # Store row
                for j in T.Parallel(n):
                    Y[bx, j] = row[j]
        return func

    y = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    func = kernel(M, N)
    func(x_2d, w, y)

    return y.view(orig_shape).to(x.dtype)
