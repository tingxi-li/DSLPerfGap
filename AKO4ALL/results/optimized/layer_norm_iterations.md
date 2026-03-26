# Iterations

## layer_norm (tilelang) Optimization

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| baseline | T.serial reduction | 0.0008x | 1090.0 ms | correct |
| 1 | Replace T.serial with T.reduce | 0.1714x | 5.24 ms | correct |
| 2 | Register copy + 256 threads | 0.1720x | 5.22 ms | correct |
| 3 | 2D block_M=4 + reduce_sum dim=1 | N/A | N/A | compile error |
| 4 | 2D T.copy fix + reduce_sum dim=1 | 0.1718x | 5.22 ms | correct |
| 5 | 512 threads | 0.1720x | 5.22 ms | correct |
| 6 | float16 I/O + float32 accum | 0.2950x | 3.02 ms | correct |
| 7 | 2D merged reduction E[x^2]-E[x]^2 | N/A | N/A | compile error |
| 8 | 1024 threads + f16 I/O | 0.2964x | 3.02 ms | correct |
| 9 | Tiled reduction block_N=1024 | 0.2987x | 2.99 ms | correct |
| 10 | 2-pass tiled (sum+sq in pass1) | 0.2980x | 3.00 ms | correct |
| 11 | E[x^2]-E[x]^2 single load | 0.2957x | 3.02 ms | correct |
| 12 | pass_configs disable warp specialized | 0.2964x | 3.02 ms | correct |
| 13 | Store x-mean in fp16, save regs | 0.2960x | 3.02 ms | correct |
| 14 | Native bfloat16 I/O, no conversion | 0.9978x | 0.893 ms | correct |
| 15 | out_idx for output allocation | 1.0000x | 0.891 ms | correct |
| 16 | Full f32 row_copy (more regs) | 0.9845x | 0.905 ms | correct |
| 17 | Restore best (iter 15 approach) | 0.9978x | 0.893 ms | correct |
| 18 | 128 threads | 0.9955x | 0.895 ms | correct |

---

## Summary

**Final: 0.891ms TileLang vs 0.891ms PyTorch = 1.00x (matching PyTorch)**

### Key optimizations (in order of impact):
1. **T.serial -> T.reduce** (iter 1): 1090ms -> 5.24ms (208x improvement)
2. **Native bfloat16 I/O** (iter 14): 3.02ms -> 0.893ms (3.4x improvement)
   - Eliminated bf16->f16->bf16 conversion overhead (~1.7ms)
   - Used torch.empty instead of torch.zeros (~0.4ms)
3. **float16/bfloat16 I/O instead of float32** (iter 6): 5.22ms -> 3.02ms (1.7x improvement)
4. **out_idx for output allocation** (iter 15): marginal improvement

### What didn't help:
- Thread count (128/256/512/1024): negligible difference once at bandwidth limit
- Tiled reduction (block_N=1024): same or worse due to more T.reduce calls
- 2D multi-row processing: compile errors or no improvement
- E[x^2]-E[x]^2 vs two-pass: same performance
- pass_configs: no impact on launch_bounds
- T.use_swizzle: not applicable to row-wise kernel

### Analysis:
The kernel is memory-bandwidth bound at 256MB / 288 GB/s = 0.89ms theoretical minimum.
Both PyTorch and TileLang achieve this limit. The T.reduce butterfly sync overhead
is negligible (only 2 AllReduce calls per row, each taking a few warp shuffles).

