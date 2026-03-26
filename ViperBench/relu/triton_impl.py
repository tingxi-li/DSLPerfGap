import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(x_ptr, out_ptr, N: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.where(x >= 0, x, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)

def relu(x):
    """Element-wise ReLU activation using Triton."""
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE), )
    relu_kernel[grid](x, out, N, BLOCK_SIZE)
    return out
