# Triton Optimization Reference

## Key Techniques
- `@triton.autotune` with multiple configs — auto-select best block sizes/warps
- Tune BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages
- Use `tl.dot()` for matmul — leverages tensor cores
- Persistent kernels for reductions — reduce kernel launch overhead
- Use `GROUP_SIZE_M` for L2 cache reuse in matmul

## Autotune Example
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

## Common Issues
- Fixed block sizes instead of autotuning
- Missing num_stages for pipelining
- Suboptimal num_warps (try 2, 4, 8)
- Not using tl.dot() for matmul patterns
- Reduction kernels with too many atomic operations
- Not using `tl.cdiv` for grid calculation
- Missing boundary masks causing incorrect results
