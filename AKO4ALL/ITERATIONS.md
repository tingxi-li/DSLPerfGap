# Iterations

## matmul (triton) Optimization

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| baseline | Fixed 64x64x64 blocks | 0.64x | 2.71 ms | -- |
| 1 | Autotune + L2 swizzle (GROUP_SIZE_M) | 1.09x | 1.62 ms | PASS |
| 2 | Expanded autotune configs | 1.08x | 1.63 ms | no improvement |
| 3 | Mask-free loads (aligned dims) | 1.07x | 1.65 ms | regression |
| 4 | Refined configs + BLOCK_K=128 | 1.08x | 1.63 ms | no improvement |
| 5 | Block pointers (tl.make_block_ptr) | 1.05x | 1.68 ms | regression |
| 6 | Split-K strategy | 1.02x | 1.73 ms | regression |
| 7 | Persistent kernel | 1.06x | 1.66 ms | regression |
| 8 | Best config set (iter-1 style) | 1.09x | 1.63 ms | confirmed best |

---

