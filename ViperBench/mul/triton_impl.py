import triton
from triton import language as tl
import torch

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("mul", "triton") or {}
except Exception:
    _TUNED = {}


@triton.jit
def mul2_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = 2 * x
    tl.store(out_ptr + offsets, output, mask=mask)


def mul(x):
    BLOCK_SIZE = _TUNED.get("BLOCK_SIZE", 1024)
    output = torch.zeros_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    return output
