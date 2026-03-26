# Iterations

## matmul (triton) Optimization

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| baseline | Fixed 64x64x64, no autotune | 0.64x | 2.72ms | - |
| 1 | Autotune (12 configs) + L2 swizzle | 1.09x | 1.62ms | BEST |
| 2 | Expanded to 18 configs + stage 5 | 1.08x | 1.63ms | no improvement |
| 3 | Removed masks (aligned dims) | 1.07x | 1.64ms | no improvement |
| 4 | Persistent kernel (48 SMs) | 1.07x | 1.66ms | no improvement |
| 5 | max_num_imprecise_acc + stage 4 configs | 1.08x | 1.63ms | no improvement |
| 6 | Clean iter-1 style with 12 configs | 1.08x | 1.63ms | converged |

---

