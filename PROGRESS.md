# TileLang Kernel Porting Progress

## Status: All 22 kernels PASS

| # | Kernel | Status | Attempts |
|---|--------|--------|----------|
| 1 | add | PASS | 1 |
| 2 | argmax | PASS | 1 |
| 3 | attention | PASS | 1 |
| 4 | batched_matmul | PASS | 1 |
| 5 | conv2d | PASS | 1 |
| 6 | cross_entropy | PASS | 1 |
| 7 | embedding | PASS | 1 |
| 8 | index_select | PASS | 1 |
| 9 | layer_norm | PASS | 1 |
| 10 | leaky_relu | PASS | 1 |
| 11 | linear_activation | PASS | 1 |
| 12 | log_softmax | PASS | 1 |
| 13 | logsumexp | PASS | 1 |
| 14 | matmul | PASS | 1 |
| 15 | matrix_transpose | PASS | 1 |
| 16 | max_reduction | PASS | 1 |
| 17 | mean_reduction | PASS | 1 |
| 18 | mul | PASS | 1 |
| 19 | relu | PASS | 1 |
| 20 | rms_norm | PASS | 1 |
| 21 | softmax | PASS | 1 |
| 22 | swiglu | PASS | 1 |

## Next Steps
- Performance benchmarking against Triton baselines
- Optimize hot kernels (attention, matmul, conv2d)
