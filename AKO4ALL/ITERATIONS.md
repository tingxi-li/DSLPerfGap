# Iterations

## matmul (triton) Optimization

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| baseline | Fixed 64x64x64 blocks | 0.64x | 2.71 ms | -- |
| 1 | Autotune + L2 swizzle (GROUP_SIZE_M) | 1.09x | 1.62 ms | PASS |

---

