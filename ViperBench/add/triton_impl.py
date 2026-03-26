import torch
import triton
import triton.language as tl

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("add", "triton") or {}
except Exception:
    _TUNED = {}

@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

def add(x, y):
    """Element-wise addition of two same-shape tensors using Triton."""
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = _TUNED.get("BLOCK_SIZE", 1024)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)
    return out
