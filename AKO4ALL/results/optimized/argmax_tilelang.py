import torch
import torch.nn as nn

# --- Original implementation inlined below ---
import torch
import tilelang
import tilelang.language as T
import math


@tilelang.jit
def _argmax_kernel_2d(M, N, bM, bN, threads=256):
    """Argmax along dim=1 for a 2D (M, N) float16 tensor.

    Each block handles bM rows. Iterate over tiles of size bN along N.
    Load tiles via T.copy (vectorized), then scan from shared memory.
    Output int64 directly to avoid post-kernel conversion.
    """
    num_tiles = T.ceildiv(N, bN)

    @T.prim_func
    def func(
        A: T.Tensor((M, N), "float16"),
        Out: T.Tensor((M,), "int64"),
    ):
        with T.Kernel(T.ceildiv(M, bM), threads=threads) as bx:
            # Shared memory tile for vectorized loading
            A_shared = T.alloc_shared((bM, bN), "float16")
            # Per-row running max value and index
            max_val = T.alloc_fragment((bM,), "float16")
            max_idx = T.alloc_fragment((bM,), "int64")

            T.clear(max_idx)
            for i in T.Parallel(bM):
                max_val[i] = T.float16(-65504.0)

            for tile in T.serial(num_tiles):
                # Vectorized bulk load: global -> shared
                T.copy(A[bx * bM, tile * bN], A_shared)

                # Scan through shared memory tile
                for j in T.serial(bN):
                    for i in T.Parallel(bM):
                        if A_shared[i, j] > max_val[i]:
                            max_val[i] = A_shared[i, j]
                            max_idx[i] = T.int64(tile * bN + j)

            T.copy(max_idx, Out[bx * bM])
    return func


# Pre-compile the kernel for the expected input size
_COMPILED_KERNEL = None
_COMPILED_KEY = None


def argmax(input_tensor, dim, keepdim=False):
    """
    Returns the indices of the maximum values across a specified dimension.
    Optimized TileLang implementation.
    """
    global _COMPILED_KERNEL, _COMPILED_KEY

    inp = input_tensor
    assert dim is not None, "dim must be specified"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim]) if dim > 0 else 1
    K = inp.numel() // M // N

    # For the common case: 2D input with dim=1 and fp16
    if K == 1 and inp.dtype == torch.float16:
        inp_2d = inp.contiguous().view(M, N)

        bM = 256
        bN = 128
        threads = 256

        # Pad M to multiple of bM
        M_pad = ((M + bM - 1) // bM) * bM

        if M_pad != M:
            inp_pad = torch.full((M_pad, N), float('-inf'), device=inp.device, dtype=torch.float16)
            inp_pad[:M, :] = inp_2d
        else:
            inp_pad = inp_2d

        out_pad = torch.empty(M_pad, dtype=torch.int64, device=inp.device)

        key = (M_pad, N, bM, bN, threads)
        if _COMPILED_KEY != key:
            _COMPILED_KERNEL = _argmax_kernel_2d(M_pad, N, bM, bN, threads=threads)
            _COMPILED_KEY = key

        _COMPILED_KERNEL(inp_pad, out_pad)

        result = out_pad[:M]
    else:
        # Fallback: use PyTorch
        return torch.argmax(inp, dim=dim, keepdim=keepdim)

    # Reshape output
    shape_list = list(shape)
    shape_list[dim] = 1
    result = result.view(shape_list)
    if not keepdim:
        result = result.squeeze(dim)
    return result

# --- End original implementation ---


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return argmax(*args)


def get_inputs():
    return [torch.randn(8192, 32768, device='cuda', dtype=torch.float16), 1]

def get_init_inputs():
    return []
