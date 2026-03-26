# Iterations

## argmax (tilelang) Optimization

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| baseline | Original T.serial kernel (fp32, 3D) | 0.11x | 16.2ms | CORRECT |
| 1 | T.reduce_max + T.Parallel index (race) | - | - | INCORRECT |
| 2 | Tiled loads (bM=4, bN=1024) + shared scan | 0.17x | 10.1ms | CORRECT |
| 3 | Direct scan, fp16 native, bM=128 | 0.23x | 7.49ms | CORRECT |
| 4 | Tiled loads (bM=64, bN=512) | 0.54x | 3.15ms | CORRECT |
| 5 | bM=32, bN=1024 | 0.29x | 5.80ms | CORRECT |
| 6 | bM=128, bN=256 | 0.70x | 2.46ms | CORRECT |
| 7 | bM=256, bN=128 (sweet spot) | 0.98x | 1.75ms | CORRECT |
| 8 | bM=512, bN=64 | 0.54x | 3.19ms | CORRECT |
| 9 | bM=256, bN=128, threads=128 | 0.98x | 1.75ms | CORRECT |
| 10 | bM=256, bN=64 | 0.83x | 2.05ms | CORRECT |
| 11 | bM=256, bN=256 (shared mem overflow) | - | - | FAILED |
| 12 | T.Pipelined (shared mem overflow) | - | - | FAILED |
| 13 | int64 output, kernel caching | 0.98x | 1.75ms | CORRECT |

---

## Key Findings

- **Best config**: bM=256, bN=128, threads=256 -> 1.75ms (0.98x PyTorch)
- **Theoretical limit**: ~1.7ms (512MB at ~300 GB/s)
- **Critical optimization**: Tiled vectorized T.copy loads + shared memory scan
- **fp16 native** halves bandwidth vs original fp32 approach
