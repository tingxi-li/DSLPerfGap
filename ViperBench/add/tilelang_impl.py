import tilelang
import tilelang.language as T
import torch

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("add", "tilelang") or {}
except Exception:
    _TUNED = {}

_block_N = _TUNED.get("block_N", 1024)
_threads = _TUNED.get("threads", 128)


@tilelang.jit
def _add_kernel(n, block_size=_block_N, dtype_str="float16"):
    @T.prim_func
    def func(
        A: T.Tensor((n,), dtype_str),
        B: T.Tensor((n,), dtype_str),
        C: T.Tensor((n,), dtype_str),
    ):
        with T.Kernel(T.ceildiv(n, block_size), threads=_threads) as bx:
            A_local = T.alloc_fragment((block_size,), dtype_str)
            B_local = T.alloc_fragment((block_size,), dtype_str)
            T.copy(A[bx * block_size], A_local)
            T.copy(B[bx * block_size], B_local)
            for i in T.Parallel(block_size):
                A_local[i] = A_local[i] + B_local[i]
            T.copy(A_local, C[bx * block_size])
    return func


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise addition of two same-shape tensors using TileLang."""
    assert x.shape == y.shape, "Shapes must match"
    N = x.numel()
    if N == 0:
        return torch.zeros_like(x)

    block_N = _block_N
    dtype_str = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(x.dtype, "float32")

    # Pad N to multiple of block_N
    padded_N = ((N + block_N - 1) // block_N) * block_N
    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)

    if padded_N != N:
        x_pad = torch.zeros(padded_N, device=x.device, dtype=x.dtype)
        y_pad = torch.zeros(padded_N, device=y.device, dtype=y.dtype)
        x_pad[:N] = x_flat
        y_pad[:N] = y_flat
    else:
        x_pad = x_flat
        y_pad = y_flat

    c_pad = torch.zeros(padded_N, device=x.device, dtype=x.dtype)

    func = _add_kernel(padded_N, dtype_str=dtype_str)
    func(x_pad, y_pad, c_pad)

    return c_pad[:N].view(x.shape)
