# Input Generation Principles for Kernel Benchmark

## Overview

Every kernel in the benchmark is evaluated across multiple **input axes**. Not all axes
apply to every kernel. This document defines which axes apply to which kernel categories,
the concrete values for each axis, and the rationale behind each choice.

---

## Global Input Axes

These are the axes that can vary across kernel inputs. Each kernel category uses a
subset of them.

| Axis | Symbol | Purpose |
|---|---|---|
| **Scale** | `S` | Element count small→large, tests launch overhead vs compute saturation |
| **Dtype** | `D` | Precision format, tests tensor core paths and memory bandwidth |
| **Sparsity/Structure** | `P` | Dense, diagonal, triangular, symmetric, banded, block-sparse |
| **Dimensionality** | `N` | 1D, 2D, 3D, 4D — tests loop nesting and memory layout |
| **Irregularity** | `I` | Non-power-of-2 dims, tests compiler boundary/masking logic |
| **Transpose** | `T` | NN, NT, TN — tests memory access pattern (row-major vs col-major) |
| **Spatial config** | `C` | Stride, padding, dilation, groups, kernel_size (conv-specific) |
| **Reduction dim** | `R` | dim=0,1,2,-1 — tests contiguous vs strided reduction |
| **Sequence length** | `L` | Short→long sequences (attention/RNN-specific) |
| **Batch size** | `B` | 1→64, tests latency-bound vs throughput-bound regimes |

---

## Axis Value Definitions

### Scale Tiers (element count)

Applied universally. The goal is to cover: kernel-launch-overhead regime,
memory-bandwidth-limited regime, and compute-saturated regime.

| Tier | Approx elements | Example 2D shape | Regime |
|---|---|---|---|
| `XS` | ~4K | (64, 64) | Launch overhead dominates |
| `S` | ~260K | (512, 512) | Memory bandwidth limited |
| `M` | ~4M | (2048, 2048) | Transitional |
| `L` | ~16M | (4096, 4096) | Compute-bound on most ops |
| `XL` | ~67M | (8192, 8192) | Fully compute-saturated |
| `XXL` | ~268M | (16384, 16384) | Stress test, register pressure |

Not all kernels need all tiers. Memory-bound ops (activation, elementwise) need
XS through L. Compute-bound ops (GEMM, attention) need S through XXL.

### Dtype Palette

| Dtype | When to use | Notes |
|---|---|---|
| `fp32` | Always | Baseline, highest precision |
| `fp16` | Always | Standard mixed-precision |
| `bf16` | Always | Preferred for LLM workloads, better dynamic range |
| `fp64` | Matmul, elementwise, reduction only | Scientific computing, rare in DL |
| `int8` | GEMM only | Quantized inference |
| `int4` | GEMM only (if hardware supports) | Weight-only quantization |
| `int16` | GEMM only (if applicable) | Mixed-precision int |

Rule: Every kernel gets {fp16, bf16, fp32}. GEMM additionally gets {int8}. 
fp64 and int4/int16 are optional extended coverage.

### Sparsity / Structure Patterns

| Pattern | Description | Applicable to |
|---|---|---|
| `dense` | Standard random init | All kernels (default) |
| `diagonal` | Only diagonal nonzero | GEMM (tests broadcast optimization) |
| `upper_triangular` | Upper triangle nonzero | GEMM (tests masking) |
| `lower_triangular` | Lower triangle nonzero | GEMM (tests masking) |
| `symmetric` | A = A^T | GEMM (tests symmetry exploitation) |
| `sparse_50` | 50% zeros, unstructured | GEMM, attention |
| `sparse_90` | 90% zeros, unstructured | GEMM, attention |
| `sparse_95` | 95% zeros, unstructured | GEMM, attention |
| `block_sparse` | 64x64 dense blocks, rest zero | GEMM (tests block-sparse paths) |
| `structured_2_4` | 2:4 NVIDIA structured sparsity | GEMM (Ampere+ tensor core path) |
| `dilated` | Dilated/strided pattern | Conv only |
| `causal_mask` | Lower-triangular attention mask | Attention only |

Rule: GEMM gets {dense, diagonal, upper_triangular, lower_triangular, symmetric,
sparse_50, sparse_90, sparse_95}. Attention gets {dense, causal_mask, sparse_90}.
Conv gets sparsity via dilated config, not input sparsity. All other categories
use dense only.

### Dimensionality

| Dims | Tensor shape | Applicable to |
|---|---|---|
| `1D` | (L,) or (B, L) | Elementwise, reduction, activation, conv1d |
| `2D` | (M, N) | GEMM, elementwise, reduction, activation |
| `3D` | (B, M, N) | Batched GEMM, normalization, activation |
| `4D` | (B, C, H, W) or (B, H, S, D) | Conv2d, attention, pooling |
| `5D` | (B, C, D, H, W) | Conv3d |

Rule: Element-wise ops and reductions sweep {1D, 2D, 3D, 4D}. GEMM uses {2D, 3D, 4D}
for unbatched/batched/multi-batched. Conv is fixed to its native dimensionality.
Attention is always 4D.

### Irregularity

For every shape config, generate a companion "irregular" variant where at least one
dimension is non-power-of-2 and not divisible by common tile sizes (32, 64, 128).

| Regular shape | Irregular companion | What it tests |
|---|---|---|
| (2048, 4096) | (1337, 4093) | Tile boundary masking |
| (8, 64, 56, 56) | (7, 64, 53, 53) | Batch and spatial boundary |
| (4, 32, 1024, 128) | (3, 32, 1337, 128) | Sequence length boundary |

Rule: For every "regular" shape config, add one irregular variant. This doubles the
shape configs but is essential — DSL performance gaps are largest on irregular shapes.

### Transpose Configs (GEMM only)

| Config | Meaning | Memory pattern |
|---|---|---|
| `NN` | C = A @ B | A row-major read, B column-major read |
| `NT` | C = A @ B^T | Both row-major read (often fastest) |
| `TN` | C = A^T @ B | Both column-major read |

Rule: Every GEMM shape gets all three transpose configs. For non-GEMM ops, transpose
does not apply.

### Reduction Dimension

| Config | Meaning | Memory pattern |
|---|---|---|
| `dim=-1` | Reduce last dimension | Contiguous memory, fast |
| `dim=0` | Reduce first dimension | Strided access, batch reduction |
| `dim=1` | Reduce middle dimension | Strided access, feature reduction |
| `dim=2` | Reduce third dimension (if 4D+) | Tests higher-dim strided reduction |

Rule: Reduction and normalization ops sweep {dim=-1, dim=0, dim=1}. Softmax
sweeps {dim=-1, dim=1}. Loss functions use their natural reduction dim.

---

## Per-Category Input Generation Specifications

### Category: `matmul` (GEMM)

**Applicable axes:** Scale, Dtype, Sparsity, Dimensionality, Irregularity, Transpose

This is the most heavily parameterized category because GEMM performance is the most
shape-sensitive.

#### Shape matrix

| Config | M | N | K | Batch | Regime | Source |
|---|---|---|---|---|---|---|
| `xs_square` | 64 | 64 | 64 | 1 | Launch overhead | Baseline |
| `s_square` | 512 | 512 | 512 | 1 | Memory BW limited | Baseline |
| `m_square` | 2048 | 2048 | 2048 | 1 | Transitional | Baseline |
| `l_square` | 4096 | 4096 | 4096 | 1 | Compute-bound | Baseline |
| `xl_square` | 8192 | 8192 | 8192 | 1 | Fully saturated | Llama-70B scale |
| `decode` | 1 | 4096 | 4096 | 1 | Memory-bound (matvec) | LLM decode |
| `small_batch_decode` | 32 | 4096 | 4096 | 1 | Small batch prefill | LLM |
| `prefill` | 2048 | 4096 | 4096 | 1 | Standard prefill | Llama-3-8B |
| `mlp_up_8B` | 2048 | 14336 | 4096 | 1 | Wide N | Llama-3-8B up_proj |
| `mlp_down_8B` | 2048 | 4096 | 14336 | 1 | Wide K | Llama-3-8B down_proj |
| `mlp_up_70B` | 2048 | 28672 | 8192 | 1 | Very wide N | Llama-3-70B |
| `moe_gate` | 2048 | 128 | 4096 | 1 | Skinny N | MoE routing |
| `moe_expert` | 256 | 14336 | 4096 | 1 | Single expert | Mixtral |
| `batched_heads` | 64 | 128 | 128 | 32 | Many small GEMMs | Attention heads |
| `large_k` | 2048 | 4096 | 32768 | 1 | Very large K | Long-context KV |
| `small_k` | 2048 | 4096 | 64 | 1 | Very small K | Narrow bottleneck |
| `irregular` | 1337 | 4093 | 4097 | 1 | All dims non-pow-2 | Stress test |

Each shape × {NN, NT, TN} × {fp16, bf16, fp32, int8} × {dense, diagonal,
upper_triangular, lower_triangular, symmetric, sparse_50, sparse_90, sparse_95}

**Prioritized subset for initial run:** All shapes × {NT} × {fp16, bf16, fp32} × {dense}
= 17 × 3 = **51 configs**

**Full sweep:** 17 × 3 × 4 × 8 = **1,632 configs** (run selectively)

---

### Category: `conv`

**Applicable axes:** Scale (via spatial dim + batch), Dtype, Spatial config, Dimensionality, Irregularity

#### Conv2d shape matrix

| Config | B | C_in | H | W | C_out | K | Stride | Pad | Dilation | Groups | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `resnet_early` | {1,8,32} | 64 | 56 | 56 | 64 | 3 | 1 | 1 | 1 | 1 | ResNet layer1 |
| `resnet_deep` | {1,8,32} | 512 | 7 | 7 | 512 | 3 | 1 | 1 | 1 | 1 | ResNet layer4 |
| `resnet_down` | {1,8,32} | 128 | 28 | 28 | 256 | 3 | 2 | 1 | 1 | 1 | Stride-2 down |
| `large_spatial` | {1,8} | 64 | 512 | 512 | 64 | 3 | 1 | 1 | 1 | 1 | Detection |
| `xlarge_spatial` | {1,4} | 32 | 1024 | 1024 | 32 | 3 | 1 | 1 | 1 | 1 | Segmentation |
| `pointwise` | {1,8,32} | 96 | 56 | 56 | 24 | 1 | 1 | 0 | 1 | 1 | MobileNet PW |
| `depthwise` | {1,8,32} | 96 | 56 | 56 | 96 | 3 | 1 | 1 | 1 | 96 | MobileNet DW |
| `dilated` | {1,8,32} | 64 | 28 | 28 | 64 | 3 | 1 | 2 | 2 | 1 | Dilated conv |
| `asymmetric_k` | {1,8,32} | 64 | 28 | 28 | 64 | (1,7) | 1 | (0,3) | 1 | 1 | Factorized |
| `patch_embed` | {1,8} | 3 | 224 | 224 | 768 | 16 | 16 | 0 | 1 | 1 | ViT patch |
| `irregular` | {1,8} | 64 | 53 | 53 | 64 | 3 | 1 | 1 | 1 | 1 | Non-pow-2 |

#### Conv1d shape matrix

Same structure but spatial dim is L only:

| Config | B | C_in | L | C_out | K | Stride | Pad | Dilation |
|---|---|---|---|---|---|---|---|---|
| `short` | {1,8,32} | 64 | 128 | 64 | 3 | 1 | 1 | 1 |
| `medium` | {1,8,32} | 64 | 1024 | 64 | 3 | 1 | 1 | 1 |
| `long` | {1,8,32} | 64 | 4096 | 64 | 3 | 1 | 1 | 1 |
| `dilated` | {1,8,32} | 64 | 1024 | 64 | 3 | 1 | 2 | 2 |

#### Conv3d shape matrix

Add depth dimension D:

| Config | B | C_in | D | H | W | C_out | K | Source |
|---|---|---|---|---|---|---|---|---|
| `small_vol` | {1,4} | 32 | 8 | 32 | 32 | 32 | 3 | Video/3D |
| `medium_vol` | {1,4} | 64 | 16 | 16 | 16 | 64 | 3 | Medical imaging |
| `large_vol` | {1,2} | 32 | 32 | 32 | 32 | 64 | 3 | Large volume |

#### Dtype: {fp16, bf16, fp32}

ConvTransposed uses the same configs with C_in/C_out swapped and output_padding added.

---

### Category: `attention`

**Applicable axes:** Scale (via seq_len), Dtype, Sparsity (causal mask), Batch, Irregularity

#### Shape matrix

| Config | B | Heads | Seq_Q | Seq_KV | Head_dim | KV_heads | Causal | Source |
|---|---|---|---|---|---|---|---|---|
| `short` | {1,4,16} | 32 | 128 | 128 | 128 | 32 | {T,F} | Short prompt, MHA |
| `medium_gqa` | {1,4,8} | 32 | 1024 | 1024 | 128 | 8 | {T,F} | Llama-3 GQA |
| `long_gqa` | {1,2,4} | 32 | 4096 | 4096 | 128 | 8 | T | Long context |
| `very_long` | {1} | 32 | 16384 | 16384 | 128 | 8 | T | Extended context |
| `ultra_long` | {1} | 32 | 32768 | 32768 | 128 | 8 | T | 32K context |
| `decode` | {1,4,16} | 32 | 1 | 2048 | 128 | 8 | T | Decode step |
| `cross_attn` | {1,4} | 32 | 512 | 2048 | 128 | 32 | F | Cross-attention |
| `vit` | {1,8,32} | 12 | 197 | 197 | 64 | 12 | F | ViT-B/16 |
| `irregular` | {1,4} | 32 | 1337 | 1337 | 128 | 8 | T | Non-pow-2 |

#### Sparsity for attention
- `dense` — full attention, no mask
- `causal` — lower-triangular causal mask
- `sparse_90` — 90% of attention weights masked (sparse attention pattern)

#### Dtype: {fp16, bf16} (fp32 optional, fp64 not supported by SDPA)

---

### Category: `normalization`

**Applicable axes:** Scale, Dtype, Dimensionality, Reduction dim, Irregularity

#### Shape matrix

Normalization kernels normalize over the last N dims. The key variable is the
ratio of normalized_shape to total tensor size.

| Config | Shape | Normalized dims | Regime | Source |
|---|---|---|---|---|
| `small_hidden` | (B, 128, 768) | last 1 (768) | Small model | GPT-2 |
| `llama_8b` | (B, 2048, 4096) | last 1 (4096) | Standard LLM | Llama-3-8B |
| `llama_70b` | (B, 2048, 8192) | last 1 (8192) | Large LLM | Llama-3-70B |
| `wide_mlp` | (B, 2048, 14336) | last 1 (14336) | MLP intermediate | Llama-3-8B |
| `vit` | (B, 197, 768) | last 1 (768) | Vision | ViT-B/16 |
| `bn_early` | (B, 64, 56, 56) | channels (64) | Conv early layer | ResNet-50 |
| `bn_deep` | (B, 2048, 7, 7) | channels (2048) | Conv deep layer | ResNet-50 |
| `irregular` | (B, 1337, 4093) | last 1 (4093) | Non-pow-2 | Stress test |

Where B ∈ {1, 8, 32}.

- **RMSNorm, LayerNorm**: use all configs except bn_* rows
- **BatchNorm, InstanceNorm, GroupNorm**: use bn_* rows and appropriately shaped tensors

#### Dtype: {fp16, bf16, fp32}

Note: BatchNorm always maintains fp32 running stats internally regardless of input dtype.

---

### Category: `activation`

**Applicable axes:** Scale, Dtype, Dimensionality, Irregularity

Activations are element-wise and memory-bound. The primary variable is total element
count (determines whether kernel launch overhead or memory BW dominates).

#### Shape matrix

| Config | Dimensionality | Shape | Elements | Regime |
|---|---|---|---|---|
| `1d_xs` | 1D | (4096,) | 4K | Launch overhead |
| `1d_l` | 1D | (4194304,) | 4M | BW saturated |
| `2d_s` | 2D | (512, 512) | 260K | Small 2D |
| `2d_l` | 2D | (4096, 4096) | 16M | Large 2D |
| `3d_m` | 3D | (8, 2048, 4096) | 67M | Typical LLM hidden |
| `3d_l` | 3D | (16, 4096, 4096) | 268M | Large batch |
| `4d` | 4D | (8, 32, 1024, 128) | 33M | Attention-like shape |
| `irregular` | 3D | (7, 1337, 4093) | 38M | Non-pow-2 |

#### Dtype: {fp16, bf16, fp32, fp64}

For `softmax` and `logsoftmax`, add reduction dim axis: {dim=-1, dim=1}.
For `swiglu` and `geglu`, input is split into two halves (gate + value), so
the input tensor has 2× the hidden dim.

---

### Category: `elementwise`

**Applicable axes:** Scale, Dtype, Dimensionality, Irregularity

Same shape matrix as activation. These are pure memory-bandwidth benchmarks.

#### Additional configs for broadcasting

| Config | A shape | B shape | Pattern |
|---|---|---|---|
| `same_shape` | (8, 2048, 4096) | (8, 2048, 4096) | No broadcast |
| `scalar_broadcast` | (8, 2048, 4096) | (1,) | Scalar broadcast |
| `row_broadcast` | (8, 2048, 4096) | (1, 1, 4096) | Row broadcast |
| `col_broadcast` | (8, 2048, 4096) | (1, 2048, 1) | Column broadcast |

---

### Category: `reduction`

**Applicable axes:** Scale, Dtype, Dimensionality, Reduction dim, Irregularity

The key insight: reduction performance depends heavily on which dim you reduce.
Reducing the last (contiguous) dim is fast. Reducing a strided dim is slow.

#### Shape matrix

| Config | Shape | Regime |
|---|---|---|
| `2d_s` | (512, 512) | Small |
| `2d_l` | (4096, 4096) | Large square |
| `3d_m` | (8, 2048, 4096) | Typical LLM |
| `3d_l` | (16, 4096, 4096) | Large |
| `wide` | (1, 1, 131072) | Single very wide row |
| `tall` | (1, 131072, 1) | Single very tall column |
| `4d` | (8, 32, 1024, 128) | Attention-like |
| `irregular` | (7, 1337, 4093) | Non-pow-2 |

#### Reduction dim sweep (for each shape)

| dim | Access pattern | Expected behavior |
|---|---|---|
| `dim=-1` | Contiguous | Fastest, warp-level reduce |
| `dim=0` | Strided (batch) | Slower, cross-block reduction |
| `dim=1` | Strided (middle) | Tests coalescing |
| `dim=2` | Strided (if 4D) | Higher-dim strided |

#### Dtype: {fp16, bf16, fp32, fp64}

For `argmax`/`argmin`, output is int64 regardless of input dtype.
For `l2_norm`, reduction is multi-step (square → sum → sqrt), test separately.

---

### Category: `loss`

**Applicable axes:** Scale, Dtype, Irregularity

#### Shape matrix

| Config | Prediction | Target | Classes | Regime | Source |
|---|---|---|---|---|---|
| `small_cls` | (32, 100) | (32,) | 100 | Small classification | CIFAR |
| `imagenet_cls` | (32, 1000) | (32,) | 1000 | ImageNet | ResNet |
| `llm_token` | (32, 32000) | (32,) | 32000 | Single-position LLM loss | Llama vocab |
| `llm_seq` | (8, 2048, 32000) | (8, 2048) | 32000 | Full-sequence LLM loss | Llama training |
| `regression` | (32, 4096) | (32, 4096) | N/A | MSE regression | Embedding loss |
| `irregular` | (7, 31999) | (7,) | 31999 | Non-pow-2 | Stress test |

For MSE/Huber: use regression config.
For CrossEntropy/KLDiv: use *_cls and llm_* configs.

#### Dtype: {fp16, bf16, fp32}

Note: CrossEntropy internally upscasts to fp32 for log-softmax stability.

---

### Category: `cumulative` (scan)

**Applicable axes:** Scale, Dtype, Dimensionality, Scan direction, Irregularity

#### Shape matrix

| Config | Shape | Scan dim | Mode | Source |
|---|---|---|---|---|
| `short` | (8, 128, 256) | dim=-1 | fwd, inclusive | Short sequence |
| `medium` | (8, 2048, 256) | dim=1 | fwd, inclusive | LLM-like |
| `long` | (8, 16384, 64) | dim=1 | fwd, inclusive | Long scan |
| `very_long` | (1, 131072, 1) | dim=1 | fwd, inclusive | SSM-scale |
| `reverse` | (8, 2048, 256) | dim=1 | reverse | Bidirectional |
| `exclusive` | (8, 2048, 256) | dim=1 | fwd, exclusive | Prefix sum |
| `irregular` | (7, 1337, 253) | dim=1 | fwd, inclusive | Non-pow-2 |

#### Dtype: {fp16, bf16, fp32, fp64}

For `cumprod`: same configs, but watch for numerical overflow at fp16 on long scans.
For `masked_cumsum`: add a random boolean mask with ~50% True values.

---

### Category: `pooling`

**Applicable axes:** Scale (via spatial dim + batch), Dtype, Spatial config, Irregularity

#### Shape matrix (2D pooling)

| Config | B | C | H | W | Pool K | Stride | Source |
|---|---|---|---|---|---|---|---|
| `resnet_maxpool` | {1,8,32} | 64 | 112 | 112 | 3 | 2 | ResNet after conv1 |
| `resnet_gap` | {1,8,32} | 2048 | 7 | 7 | 7 | 1 | ResNet global avg pool |
| `large_spatial` | {1,8} | 64 | 256 | 256 | 3 | 2 | Detection |
| `irregular` | {1,8} | 64 | 53 | 53 | 3 | 2 | Non-pow-2 |

Adapt for 1D (drop W) and 3D (add D).

#### Dtype: {fp16, bf16, fp32}

---

### Category: `fused_matmul`, `fused_gemm`

**Applicable axes:** Scale, Dtype, Irregularity

These use the GEMM shape matrix but with the shapes relevant to their dominant op.

#### Principle

Use a **subset** of the GEMM shapes that cover the three key regimes:

| Config | Shape | Why |
|---|---|---|
| `decode` | M=1, N=4096, K=4096 | Latency-bound |
| `prefill` | M=2048, N=4096, K=4096 | Compute-bound |
| `mlp` | M=2048, N=14336, K=4096 | Wide, MLP-like |
| `irregular` | M=1337, N=4093, K=4097 | Boundary test |

For `fused_gemm` (nn.Linear-based): `in_features` and `out_features` are fixed at init.
Generate inputs with varying batch/seq dims: B×S ∈ {1, 32, 2048} for the M dimension.

#### Dtype: {fp16, bf16, fp32}

---

### Category: `fused_conv`, `fused_convtranspose`

**Applicable axes:** Scale (via batch + spatial), Dtype, Irregularity

Use the **conv shape matrix** subset:

| Config | Source |
|---|---|
| `resnet_early` | (B, 64, 56, 56) → conv → fused ops |
| `resnet_deep` | (B, 512, 7, 7) → conv → fused ops |
| `large_spatial` | (B, 64, 512, 512) → conv → fused ops |
| `irregular` | (B, 64, 53, 53) → conv → fused ops |

With B ∈ {1, 8, 32}.

For fused_convtranspose: swap C_in/C_out, add output_padding.

#### Dtype: {fp16, bf16, fp32}

---

### Category: `normalization`

See dedicated section above (Section "Category: normalization").

---

### Category: `embedding`

**Applicable axes:** Scale (vocab size, seq length), Dtype

| Config | Vocab | Seq_len | Embed_dim | Batch |
|---|---|---|---|---|
| `small` | 1000 | 128 | 768 | 32 |
| `llm` | 32000 | 2048 | 4096 | 8 |
| `large_vocab` | 128000 | 2048 | 4096 | 4 |

Input indices: int64. Embedding weights: {fp16, bf16, fp32}.

---

### Category: `positional_encoding` (RoPE)

**Applicable axes:** Same as attention shape matrix

RoPE operates on Q/K tensors of shape (B, H, S, D). Use the attention shape matrix
from the attention section. The key variables are seq_len and head_dim.

#### Dtype: {fp16, bf16, fp32}

---

### Category: `model_cnn`, `model_transformer`, `model_rnn`, `model_other`

**Applicable axes:** Batch size, Dtype

These are model-level tasks with fixed architecture shapes. The only swept axes are:

| Axis | Values |
|---|---|
| Batch size | {1, 8, 32} (or {1, 4, 16} for large models) |
| Dtype | {fp16, bf16, fp32} |
| Input | Model-canonical (e.g., (B,3,224,224) for CNNs) |

For transformer decoder blocks, additionally sweep seq_len ∈ {128, 1024, 4096}.

---

### Category: `specialized` (jagged ops, mamba, etc.)

These have kernel-specific input formats that don't fit the standard axes.

**Jagged ops** (jagged_softmax, jagged_mean, jagged_sum, jagged_layer_norm):
- Variable-length sequences packed contiguously
- Lengths sampled from: {64, 128, 256, 512, 1024, 2048} (mixed within batch)
- Total batch tokens: {4K, 32K, 256K}

**Mamba ops** (mamba2_chunk_scan, mamba2_chunk_state):
- Use Mamba-2 paper's default configs: d_model=2048, d_state=128, chunk_size=256
- Sweep seq_len ∈ {1024, 4096, 16384}

**launch_latency**: No data input, measures kernel dispatch overhead.

---

## Input Generation Implementation Notes

### Random initialization

- Float tensors: `torch.randn(shape, dtype=dtype, device='cuda')`
- Int tensors: `torch.randint(0, range, shape, dtype=dtype, device='cuda')`
- Sparse tensors: Generate dense, then zero out entries randomly

### Structured matrix generation

```python
def make_diagonal(n, dtype):
    return torch.diag(torch.randn(n, dtype=dtype, device='cuda'))

def make_upper_triangular(m, n, dtype):
    return torch.triu(torch.randn(m, n, dtype=dtype, device='cuda'))

def make_lower_triangular(m, n, dtype):
    return torch.tril(torch.randn(m, n, dtype=dtype, device='cuda'))

def make_symmetric(n, dtype):
    A = torch.randn(n, n, dtype=dtype, device='cuda')
    return (A + A.T) / 2

def make_sparse(shape, sparsity, dtype):
    A = torch.randn(shape, dtype=dtype, device='cuda')
    mask = torch.rand(shape, device='cuda') > sparsity
    return A * mask
```

### Warm-up and timing

- Always run 10+ warm-up iterations before timing
- Use `torch.cuda.synchronize()` before and after timed region
- Use `torch.cuda.Event` for precise GPU timing
- Report median of 100+ iterations (not mean, to avoid outlier skew)
- Clear L2 cache between iterations if measuring cold-cache performance

### Reproducibility

- Set `torch.manual_seed(42)` and `torch.cuda.manual_seed(42)` before each config
- Record exact GPU model, driver version, CUDA version, PyTorch version
- Record GPU clock speed (lock clocks for consistent results: `nvidia-smi -lgc MIN,MAX`)

---

## Summary: Input Axis Applicability Matrix

| Category | Scale | Dtype | Sparsity | Dims | Irregular | Transpose | Spatial cfg | Reduce dim | Seq len | Batch |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | | | ✓ |
| conv | ✓ | ✓ | | | ✓ | | ✓ | | | ✓ |
| attention | ✓ | ✓ | ✓ | | ✓ | | | | ✓ | ✓ |
| normalization | ✓ | ✓ | | | ✓ | | | | | ✓ |
| activation | ✓ | ✓ | | ✓ | ✓ | | | | | |
| elementwise | ✓ | ✓ | | ✓ | ✓ | | | | | |
| reduction | ✓ | ✓ | | ✓ | ✓ | | | ✓ | | |
| loss | ✓ | ✓ | | | ✓ | | | | | |
| cumulative | ✓ | ✓ | | | ✓ | | | | | |
| pooling | ✓ | ✓ | | | ✓ | | ✓ | | | ✓ |
| fused_matmul | ✓ | ✓ | | | ✓ | | | | | ✓ |
| fused_conv | ✓ | ✓ | | | ✓ | | | | | ✓ |
| model_* | | ✓ | | | | | | | ✓* | ✓ |
| specialized | ✓ | ✓ | | | | | | | ✓ | |

*seq_len only for transformer/RNN models