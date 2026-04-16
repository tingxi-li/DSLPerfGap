# ViperBench Dataset Construction Specification

This document describes how the raw kernel collection was transformed into
the `ViperBench/kernels/` directory structure defined in `docs/SPEC.md`.

> **Note:** The original `KernelBench_dedup/` directory has been removed and
> superseded by `kernel_samples/`, which contains the consolidated source
> kernels organized by category.

-----

## 1. Source Material

The original sources (now in `kernel_samples/`) were:

| Source | Original Path | Count | Format |
|--------|---------------|-------|--------|
| KernelBench L1 | `KernelBench_dedup/level1/` | 100 | `Model` class + `get_inputs()` + `get_init_inputs()` |
| KernelBench L2 | `KernelBench_dedup/level2/` | 99 | Same |
| KernelBench L3 | `KernelBench_dedup/level3/` | 59 | Same |
| TritonBench | `KernelBench_dedup/tritonbench/` | 36 | `operator.py` with Triton kernels |
| Categories | `KernelBench_dedup/categories/categories.json` | — | Category → kernel list mapping |

**Total source kernels:** 294 (after dedup)

-----

## 2. Consolidation Rules

The raw dataset contains many near-variants of the same logical operation
(e.g., 18 matmul variants, 35 conv variants). ViperBench consolidates these
into a single kernel directory per logical operation, with variants expressed
as axes in `input_config.json`.

### 2.1 Matmul Consolidation

**Source:** 18 level1 matmul kernels + 4 tritonbench GEMM operators = 22 raw kernels.

**Target:** 1 kernel directory: `kernels/matmul/`

| Raw kernel | Maps to |
|------------|---------|
| `1_square_matrix_multiplication` | shape: `{xs_square, s_square, m_square, ...}` |
| `2_standard_matrix_multiplication` | shape: rectangular configs |
| `3_batched_matrix_multiplication` | batch > 1 configs |
| `4_matrix_vector_multiplication` | shape: `decode` (M=1) |
| `5_matrix_scalar_multiplication` | **Separate kernel:** `kernels/scalar_mul/` (elementwise, not GEMM) |
| `6_matmul_with_large_k_dimension` | shape: `large_k` |
| `7_matmul_with_small_k_dimension` | shape: `small_k` |
| `8_matmul_with_irregular_shapes` | shape: `irregular` |
| `9_tall_skinny_matrix_multiplication` | shape: `decode`, `moe_gate` |
| `10_3d_tensor_matrix_multiplication` | dims: 3D (batch dim present) |
| `11_4d_tensor_matrix_multiplication` | dims: 4D (`batched_heads` config) |
| `12_matmul_with_diagonal_matrices` | structure: `diagonal` |
| `13_matmul_for_symmetric_matrices` | structure: `symmetric` |
| `14_matmul_for_upper_triangular_matrices` | structure: `upper_triangular` |
| `15_matmul_for_lower_triangular_matrices` | structure: `lower_triangular` |
| `16_matmul_with_transposed_a` | transpose: `TN` |
| `17_matmul_with_transposed_b` | transpose: `NT` |
| `18_matmul_with_transposed_both` | transpose: `TT` (add to axis) |
| `tritonbench/addmm` | Absorbed into matmul (matmul + bias add) |
| `tritonbench/gemm` | Absorbed into matmul |
| `tritonbench/grouped_gemm` | **Separate kernel:** `kernels/grouped_gemm/` |
| `tritonbench/gather_gemv` | **Separate kernel:** `kernels/gather_gemv/` |

**Result:** 22 raw → 1 `matmul` + 3 separate = **4 kernel dirs.**

The single `matmul/reference.py` implements:
```python
def reference(inputs):
    A, B = inputs["A"], inputs["B"]
    return {"C": torch.matmul(A, B)}
```

All shape, transpose, structure, and batch variations are expressed in
`matmul/input_config.json` per SPEC Section 5.1.

### 2.2 Convolution Consolidation

**Source:** 35 level1 conv kernels.

**Target:** 7 kernel directories.

| Target kernel | Raw kernels absorbed | Distinguishing feature |
|---------------|---------------------|----------------------|
| `kernels/conv1d/` | 67, 76 | 1D convolution, stride/dilation via spatial config axis |
| `kernels/conv2d/` | 50, 55, 56, 62, 63, 80 | Standard 2D conv, all kernel/input shape combos via axes |
| `kernels/conv3d/` | 54, 59, 60, 66 | Standard 3D conv |
| `kernels/conv_transposed_1d/` | 64, 74, 79 | Transposed 1D, dilation/stride via axes |
| `kernels/conv_transposed_2d/` | 57, 65, 69, 71, 75, 78, 81 | Transposed 2D |
| `kernels/conv_transposed_3d/` | 58, 61, 68, 70, 72, 73, 77 | Transposed 3D |
| `kernels/conv_depthwise_2d/` | 82, 83, 84, 85, 86, 87 | Depthwise, separable, pointwise |

**Result:** 35 raw → **7 kernel dirs.**

Each `reference.py` uses `torch.nn.functional.conv*d` with parameters drawn
from `input_config.json`. Stride, padding, dilation, groups, and
kernel-size asymmetry are all axes, not separate kernels.

### 2.3 Activation Consolidation

**Source:** 17 activation kernels (15 level1 + 2 tritonbench).

**Target:** 17 kernel directories (1:1 — each activation function is a
distinct mathematical operation).

| Target kernel | Source |
|---------------|--------|
| `kernels/relu/` | level1/19_relu |
| `kernels/leaky_relu/` | level1/20_leakyrelu |
| `kernels/sigmoid/` | level1/21_sigmoid |
| `kernels/tanh/` | level1/22_tanh |
| `kernels/softmax/` | level1/23_softmax |
| `kernels/log_softmax/` | level1/24_logsoftmax |
| `kernels/swish/` | level1/25_swish |
| `kernels/gelu/` | level1/26_gelu |
| `kernels/selu/` | level1/27_selu |
| `kernels/hard_sigmoid/` | level1/28_hardsigmoid |
| `kernels/softplus/` | level1/29_softplus |
| `kernels/softsign/` | level1/30_softsign |
| `kernels/elu/` | level1/31_elu |
| `kernels/hard_tanh/` | level1/32_hardtanh |
| `kernels/mingpt_gelu/` | level1/88_mingptnewgelu |
| `kernels/geglu/` | tritonbench/geglu |
| `kernels/swiglu/` | tritonbench/swiglu |

Each shares the activation input config (SPEC Section 5.5) with shape, dtype,
dimensionality, and irregularity axes.

### 2.4 Normalization Consolidation

**Source:** 10 normalization kernels (8 level1 + 2 tritonbench).

**Target:** 8 kernel directories.

| Target kernel | Sources absorbed |
|---------------|-----------------|
| `kernels/batch_norm/` | level1/33_batchnorm |
| `kernels/instance_norm/` | level1/34_instancenorm |
| `kernels/group_norm/` | level1/35_groupnorm |
| `kernels/rms_norm/` | level1/36_rmsnorm + tritonbench/rms_norm |
| `kernels/frobenius_norm/` | level1/37_frobeniusnorm |
| `kernels/l1_norm/` | level1/38_l1norm |
| `kernels/l2_norm/` | level1/39_l2norm |
| `kernels/layer_norm/` | level1/40_layernorm + tritonbench/layer_norm |

**Result:** 10 raw → **8 kernel dirs** (2 tritonbench absorbed as DSL impls).

For `rms_norm` and `layer_norm`, the tritonbench Triton code becomes the
initial `triton_impl.py` instead of a stub.

### 2.5 Pooling Consolidation

**Source:** 6 level1 pooling kernels.

**Target:** 6 kernel directories (1:1 — MaxPool and AvgPool at each
dimensionality are distinct operations with different computation).

| Target kernel | Source |
|---------------|--------|
| `kernels/max_pool_1d/` | level1/41 |
| `kernels/max_pool_2d/` | level1/42 |
| `kernels/max_pool_3d/` | level1/43 |
| `kernels/avg_pool_1d/` | level1/44 |
| `kernels/avg_pool_2d/` | level1/45 |
| `kernels/avg_pool_3d/` | level1/46 |

### 2.6 Reduction Consolidation

**Source:** 7 reduction kernels (6 level1 + 1 tritonbench).

**Target:** 6 kernel directories.

| Target kernel | Sources |
|---------------|---------|
| `kernels/sum_reduction/` | level1/47 + tritonbench/sum |
| `kernels/mean_reduction/` | level1/48 |
| `kernels/max_reduction/` | level1/49 |
| `kernels/min_reduction/` | level1/53 |
| `kernels/argmax/` | level1/51 |
| `kernels/argmin/` | level1/52 |

Reduction dim is an axis in `input_config.json`, not a separate kernel.

### 2.7 Loss Consolidation

**Source:** 9 loss kernels (6 level1 + 3 tritonbench).

**Target:** 8 kernel directories.

| Target kernel | Sources |
|---------------|---------|
| `kernels/mse_loss/` | level1/94 |
| `kernels/cross_entropy/` | level1/95 + tritonbench/cross_entropy |
| `kernels/huber_loss/` | level1/96 |
| `kernels/kl_div/` | level1/98 + tritonbench/kl_div |
| `kernels/triplet_margin_loss/` | level1/99 |
| `kernels/hinge_loss/` | level1/100 |
| `kernels/jsd/` | tritonbench/jsd |
| `kernels/fused_linear_cross_entropy/` | tritonbench/fused_linear_cross_entropy |

`tritonbench/fused_linear_jsd` is absorbed into `jsd` as a variant.

### 2.8 Attention Consolidation

**Source:** 10 attention kernels (1 level1 + 9 tritonbench).

**Target:** 5 kernel directories.

| Target kernel | Sources absorbed |
|---------------|-----------------|
| `kernels/sdpa/` | level1/97_scaleddotproductattention, tritonbench/flash_attention, tritonbench/template_attention |
| `kernels/flex_attention/` | tritonbench/flex_attention, tritonbench/custom_shape_attentions |
| `kernels/fp8_attention/` | tritonbench/fp8_attention |
| `kernels/decoding_attention/` | tritonbench/decoding_attention |
| `kernels/ragged_attention/` | tritonbench/ragged_attention |

`tritonbench/gdpa` absorbed into `sdpa` (variant).
`tritonbench/blackwell_attentions` is arch-specific → `benchmark_tier: "arch_specific"`.

### 2.9 Cumulative Consolidation

**Source:** 5 level1 cumulative kernels.

**Target:** 3 kernel directories.

| Target kernel | Sources | Mode axis values |
|---------------|---------|-----------------|
| `kernels/cumsum/` | 89_cumsum, 91_cumsum_reverse, 92_cumsum_exclusive | mode: {forward_inclusive, reverse, exclusive} |
| `kernels/cumprod/` | 90_cumprod | — |
| `kernels/masked_cumsum/` | 93_masked_cumsum | — |

Forward/reverse/exclusive are scan-mode axis values, not separate kernels.

### 2.10 Quantization Kernels

**Source:** 11 tritonbench quantization operators.

**Target:** 6 kernel directories.

| Target kernel | Sources absorbed |
|---------------|-----------------|
| `kernels/fp8_gemm/` | fp8_gemm, fp8_gemm_blockwise, fp8_gemm_rowwise, fp8_gemm_rowwise_grouped, fp8_fused_quant_gemm_rowwise |
| `kernels/int4_gemm/` | int4_gemm |
| `kernels/mixed_gemm/` | mixed_gemm, bf16xint16_gemm |
| `kernels/nvfp4_gemm/` | nvfp4_gemm |
| `kernels/fp32_to_mx4/` | fp32_to_mx4 |
| `kernels/mx4_to_fp32/` | mx4_to_fp32 |

FP8 GEMM variants (blockwise, rowwise, grouped, fused-quant) become a
`quantization_mode` axis within `fp8_gemm/input_config.json`.

### 2.11 Elementwise Consolidation

**Source:** 2 tritonbench elementwise + level1/5_matrix_scalar_multiplication.

**Target:** 3 kernel directories.

| Target kernel | Source |
|---------------|--------|
| `kernels/vector_add/` | tritonbench/vector_add |
| `kernels/vector_exp/` | tritonbench/vector_exp |
| `kernels/scalar_mul/` | level1/5_matrix_scalar_multiplication |

### 2.12 Specialized Kernels

**Source:** 10 tritonbench specialized operators.

**Target:** 8 kernel directories.

| Target kernel | Sources |
|---------------|---------|
| `kernels/rope/` | tritonbench/rope |
| `kernels/jagged_mean/` | tritonbench/jagged_mean |
| `kernels/jagged_softmax/` | tritonbench/jagged_softmax |
| `kernels/jagged_sum/` | tritonbench/jagged_sum |
| `kernels/jagged_layer_norm/` | tritonbench/jagged_layer_norm |
| `kernels/mamba2_chunk_scan/` | tritonbench/mamba2_chunk_scan |
| `kernels/mamba2_chunk_state/` | tritonbench/mamba2_chunk_state |
| `kernels/welford/` | tritonbench/welford |

`tritonbench/gdn_fwd_h` → absorbed into a specialized kernel or dropped
(niche op). `tritonbench/launch_latency` → not a computational kernel;
used as a harness diagnostic, not a benchmark entry.

### 2.13 Embedding / Dropout

| Target kernel | Source |
|---------------|--------|
| `kernels/embedding/` | tritonbench/embedding |
| `kernels/low_mem_dropout/` | tritonbench/low_mem_dropout |

### 2.14 Fused Ops (Level 2)

**Source:** 99 level2 fused op chains.

**Target:** 99 kernel directories (1:1 — each fused chain is a unique
computational graph).

Directory naming: `kernels/fused_<primary_op>_<N>/` where N is the original
level2 number.

Examples:
- `kernels/fused_conv2d_1/` ← level2/1_conv2d_relu_biasadd
- `kernels/fused_gemm_12/` ← level2/12_gemm_multiply_leakyrelu
- `kernels/fused_matmul_9/` ← level2/9_matmul_subtract_multiply_relu

Each uses the fused-op input config (SPEC Section 5.11 / 5.12).

### 2.15 Model-Level (Level 3)

**Source:** 59 level3 model architectures.

**Target:** 59 kernel directories (1:1 — each is a distinct architecture).

Directory naming: `kernels/model_<name>/` preserving the original name.

Examples:
- `kernels/model_resnet18/` ← level3/9_resnet18
- `kernels/model_vit/` ← level3/28_visiontransformer
- `kernels/model_lstm/` ← level3/35_lstm

Each uses the model-level input config (SPEC Section 5.15).

-----

## 3. Consolidation Summary

| Category | Raw kernels | Target kernel dirs | Consolidation ratio |
|----------|------------|-------------------|-------------------|
| matmul | 22 | 4 | 5.5:1 |
| conv | 35 | 7 | 5.0:1 |
| activation | 17 | 17 | 1:1 |
| normalization | 10 | 8 | 1.25:1 |
| pooling | 6 | 6 | 1:1 |
| reduction | 7 | 6 | 1.17:1 |
| loss | 9 | 8 | 1.13:1 |
| attention | 10 | 5 | 2:1 |
| cumulative | 5 | 3 | 1.67:1 |
| quantization | 11 | 6 | 1.83:1 |
| elementwise | 3 | 3 | 1:1 |
| specialized | 10 | 8 | 1.25:1 |
| embedding | 1 | 1 | 1:1 |
| dropout | 1 | 1 | 1:1 |
| fused_ops (L2) | 99 | 99 | 1:1 |
| models (L3) | 59 | 59 | 1:1 |
| **Total** | **305** | **241** | **1.27:1** |

The primary consolidation wins come from matmul (22→4), conv (35→7), and
attention (10→5), where shape/transpose/structure variants collapse into
input config axes.

-----

## 4. Construction Process

### 4.1 Automated Steps (scriptable)

For each target kernel directory:

1. **Create directory:** `kernels/<name>/`
2. **Write `metadata.json`:** Populate from `categories.json` + source file
   docstrings.
3. **Write `input_config.json`:** Use the per-category templates from SPEC
   Sections 5.1–5.16. Map raw kernel parameters to axis values.
4. **Write `reference.py`:** Extract `Model.forward()` from the source
   `pytorch_impl.py`. Adapt to the `reference(inputs: dict) -> dict` signature.
   For consolidated kernels, write a unified reference that handles all axis
   values.
5. **Write `triton_impl.py` stub:** Default `NOT_IMPLEMENTED = True`.
   For kernels with tritonbench source, extract the Triton kernel into the
   `kernel(inputs: dict) -> dict` format instead.
6. **Write `tilelang_impl.py` stub:** Default `NOT_IMPLEMENTED = True`.

### 4.2 Reference Extraction Rules

**Level 1/2/3 (KernelBench format):**

Source format:
```python
class Model(nn.Module):
    def __init__(self, ...):
        self.layer = nn.Something(...)
    def forward(self, x):
        return self.layer(x)

def get_inputs():
    return [torch.randn(...)]
def get_init_inputs():
    return [param1, param2]
```

Target format:
```python
def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    # Reconstruct the computation without nn.Module state
    # For stateful ops (conv, norm), create layer inline
    layer = torch.nn.Conv2d(...)
    layer.weight.data = inputs["weight"]
    layer.bias.data = inputs["bias"]
    return {"output": layer(x)}
```

**Key transformations:**
- `nn.Module` parameters (weights, biases) become explicit input tensors.
- Layer construction uses parameters from `input_config.json`, not hardcoded.
- Output is a named dict, not a raw tensor.

**Exception — model-level kernels (L3):** These keep `nn.Module` internally
since they have many layers. The reference wraps the Model class:
```python
def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model = _build_model()  # cached
    return {"output": model(inputs["input"])}
```

**TritonBench format:**

Source: `operator.py` with various methods. Extract the `forward()` or
`execute()` method. The Triton kernel is in the same file or a companion.

### 4.3 Input Config Generation Rules

For each category, use the template from SPEC Section 5. Map the source
kernel's hardcoded parameters to the appropriate axis:

| Source parameter | Maps to axis | Example |
|-----------------|-------------|---------|
| Shape dimensions | `params.shape` entries | `M=64, N=64, K=64` → `xs_square` |
| `.half()` / dtype | `params.dtype` | `["fp16", "bf16", "fp32"]` |
| `stride=2` | `params.spatial.stride` | `[1, 2]` |
| `padding=1` | `params.spatial.padding` | `[0, 1, 2]` |
| `dilation=2` | `params.spatial.dilation` | `[1, 2]` |
| `groups=C_in` | `params.spatial.groups` | `[1, "depthwise"]` |
| `A.T @ B` | `params.transpose` | `["NN", "NT", "TN"]` |
| `torch.diag(A)` | `params.structure` | `["dense", "diagonal"]` |
| `dim=1` | `params.reduction_dim` | `[-1, 0, 1, 2]` |

### 4.4 Triton Implementation Extraction

For tritonbench operators that have Triton source:

1. Read `operator.py` from the tritonbench directory.
2. Extract the `@triton.jit` kernel function(s).
3. Extract the Python launch wrapper.
4. Adapt to `kernel(inputs: dict) -> dict` signature.
5. Set `NOT_IMPLEMENTED = False`.

For kernels without tritonbench source: write stub with
`NOT_IMPLEMENTED = True`.

-----

## 5. Validation Checklist

After construction, verify:

- [ ] Every `kernels/*/` has exactly: `metadata.json`, `input_config.json`,
      `reference.py`, `triton_impl.py`, `tilelang_impl.py`
- [ ] Every `metadata.json` has all required fields per SPEC Section 3.1
- [ ] Every `input_config.json` has valid `params`, `sweep_mode`,
      `priority_sweep`, `correctness`, `profiling` sections
- [ ] Every `reference.py` defines `reference(dict) -> dict`
- [ ] Every `triton_impl.py` and `tilelang_impl.py` defines `kernel(dict) -> dict`
      (or has `NOT_IMPLEMENTED = True`)
- [ ] `python -m viperbench.runner --list` shows 241 kernels
- [ ] `python -m viperbench.runner --all --correctness-only --sweep prioritized`
      passes for all reference implementations
- [ ] No two kernel directories contain functionally identical computations
- [ ] All tritonbench operators with Triton source have `NOT_IMPLEMENTED = False`
      in `triton_impl.py`
- [ ] Kernel count by category matches Section 3 summary table

-----

## 6. File Generation Order

To minimize circular dependencies during construction:

1. **Phase 1 — Primitive ops** (67 kernel dirs): matmul, conv, activation,
   normalization, pooling, reduction, loss, cumulative, elementwise,
   embedding, dropout
2. **Phase 2 — Advanced ops** (19 kernel dirs): attention, quantization,
   specialized, rope
3. **Phase 3 — Fused ops** (99 kernel dirs): All level2 fused chains
4. **Phase 4 — Models** (59 kernel dirs): All level3 model architectures
5. **Phase 5 — Validation**: Run correctness sweep, fix any issues
