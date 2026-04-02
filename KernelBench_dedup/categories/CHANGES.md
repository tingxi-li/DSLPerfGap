# Categorization Changes from Original categories.json

20 corrections applied based on implementation-level verification of each kernel's `forward()` method.

## Level 1 (1 change)

| Kernel | Old Category | New Category | Reason |
|--------|-------------|--------------|--------|
| level1/5_matrix_scalar_multiplication | matmul | elementwise | Uses `A * s` (scalar multiply), not torch.matmul |

## Level 2 (10 changes)

| Kernel | Old Category | New Category | Reason |
|--------|-------------|--------------|--------|
| level2/9_matmul_subtract_multiply_relu | fused_matmul | fused_gemm | Uses nn.Linear (GEMM), not torch.matmul |
| level2/14_gemm_divide_sum_scaling | fused_gemm | fused_matmul | Uses torch.matmul(), not nn.Linear |
| level2/18_matmul_sum_max_avgpool_logsumexp_logsumexp | fused_matmul | fused_gemm | Uses nn.Linear |
| level2/29_matmul_mish_mish | fused_matmul | fused_gemm | Uses nn.Linear |
| level2/56_matmul_sigmoid_sum | fused_matmul | fused_gemm | Uses nn.Linear |
| level2/62_matmul_groupnorm_leakyrelu_sum | fused_matmul | fused_gemm | Uses nn.Linear |
| level2/68_matmul_min_subtract | fused_matmul | fused_gemm | Uses nn.Linear |
| level2/86_matmul_divide_gelu | fused_matmul | fused_gemm | Uses nn.Linear |
| level2/99_matmul_gelu_softmax | fused_matmul | fused_gemm | Uses nn.Linear |

Note: kernel names say "matmul" but implementations use `nn.Linear` (GEMM with learned weights).

## Level 3 (4 changes)

| Kernel | Old Category | New Category | Reason |
|--------|-------------|--------------|--------|
| level3/1_mlp | model_cnn | model_other | Pure MLP (nn.Linear only), no convolutions |
| level3/2_shallowwidemlp | model_cnn | model_other | Pure MLP, no convolutions |
| level3/3_deepnarrowmlp | model_cnn | model_other | Pure MLP, no convolutions |
| level3/45_unetsoftmax | model_other | model_cnn | Full U-Net with Conv2d encoder + ConvTranspose2d decoder |

## Tritonbench (5 changes)

| Kernel | Old Category | New Category | Reason |
|--------|-------------|--------------|--------|
| tritonbench/rope | specialized | embedding | Rotary Position Embeddings — positional embedding op |
| tritonbench/welford | specialized | normalization | Implements F.layer_norm (Welford is the underlying algorithm) |
| tritonbench/fused_linear_cross_entropy | fused_matmul | loss | Primary output is cross-entropy loss value |
| tritonbench/fused_linear_jsd | fused_matmul | loss | Primary output is JSD loss value |
| tritonbench/mixed_gemm | quantization | specialized | Mixed-precision bf16 GEMM, not a quantization format |
