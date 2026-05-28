# Iterations

## conv2d (triton) Optimization

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| baseline | Original kernel | 0.31x | 32.1ms | baseline |
| 1 | Enable fp16 + autotune | 0.38x | 26.0ms | pass |
| 2 | Implicit GEMM formulation | 0.55x | 18.3ms | pass |
| 3 | L2 swizzle + tl.dot acc | 0.70x | 14.2ms | pass |
| 4 | Separate kh/kw loops | 0.54x | 18.5ms | regression |
| 5 | Expanded autotune configs | 0.71x | 14.3ms | pass |
| 6 | NHWC layout | 0.34x | 29.2ms | regression |
| 7 | IC-first K ordering | 0.70x | 14.3ms | pass |
| 8 | Explicit im2col + matmul | 0.44x | 22.7ms | regression |
| 9 | Weight cache | -10x | FAIL | fail (stale cache) |
| 10 | torch.unfold + batched mm | 0.17x | 59.0ms | regression |
| 11 | Cleaned implicit GEMM | 0.70x | 14.3ms | pass |
| 12 | 3D grid (batch dim) | 0.71x | 14.3ms | pass |
| 13 | **Padded input (no bounds check)** | **0.80x** | **12.5ms** | **best** |
| 14 | Padded + native weight | 0.48x | 20.8ms | regression |
| 15 | More autotune configs | 0.80x | 12.5ms | pass |

---

## Best Result
- **Runtime**: 12.5ms (min 11.7ms)
- **Speedup**: 0.80x vs cuDNN (10.0ms)
- **Approach**: Implicit GEMM with padded input, pre-transposed weight, fp16 tensor cores, L2 swizzle, autotuning
