# #1787 ICSE Research Track

> *"An Empirical Study of GPU Kernel Performance Gaps in Modern Domain-Specific Languages"*

This artifact supports all claims in the paper. It contains the benchmark suite, the
iterative kernel-optimization loop (AKO4ALL), all rebuttal experiments, and the LaTeX paper source.

**E_lib %** = library efficiency (PyTorch / cuBLAS / cuDNN = 100%; higher is better).

---

## Repository Structure

```
ASE-GPUDSL-ARTIFACT/
├── ViperBench/                        # benchmark suite — 22 kernels (PyTorch / Triton / TileLang)
│   ├── <kernel>/                      # pytorch_impl.py · triton_impl.py · tilelang_impl.py · test.py
│   ├── benchmark*.py                  # latency + peak-memory sweep
│   ├── run_all.py                     # correctness runner
│   └── results/
│       ├── profile.A100-SXM4-40GB.csv
│       ├── profile.GH200-480GB.csv
│       ├── profile.A100-PCIE-40GB.csv
│       ├── profile.H100-80GB-HBM3.csv
│       └── slow_kernels.csv           # kernels where DSL < PyTorch at large shape
├── AKO4ALL/                           # Iterative kernel-optimization loop
│   ├── prepare_kernel.py              # benchmark suite → KernelBench format export
│   ├── TASK.md                        # Protocol governing the optimization agent
│   ├── HINTS.md                       # Constraints (no language switching, etc.)
│   └── results/
│       ├── optimization_results.csv   # Per-campaign best speedups (5 campaigns)
│       └── optimized/                 # Per-kernel optimized source + iteration logs
├── experiments/                       # Root-cause + rebuttal experiments
│   ├── exp_*.py                       # One script per experiment
│   ├── ncu_counters.sh                # Nsight Compute counter sweep driver
│   ├── consolidate_ncu.py             # Roll-up ncu CSVs → ncu_summary.csv
│   ├── A100_H100_RUNBOOK.md           # Cross-arch replay instructions
│   └── results/
│       ├── NVIDIA_A100-SXM4-40GB/    # Primary A100 evidence
│       ├── NVIDIA_GH200_480GB/        # Primary GH200 evidence
│       ├── NVIDIA_A100-PCIE-40GB/     # Cross-arch form-factor control
│       ├── NVIDIA_H100_80GB_HBM3/     # Cross-arch Hopper control
│       └── cross_arch/                # Aggregated tables + generator script
└── paper-latex-project/               # LaTeX source; build with pdflatex + bibtex
```

---

## Kernel Suite Provenance

The 22-kernel benchmark suite covers five operator categories.
Triton implementations are drawn from TritonBench except `layer_norm` (TorchInductor reference).
All TileLang implementations are custom re-implementations of the same operators.

| Kernel | Category | Triton Source | TileLang Source |
|---|---|---|---|
| `matmul` | GEMM | TritonBench | Custom |
| `batched_matmul` | GEMM | TritonBench | Custom |
| `linear_activation` | GEMM | TritonBench | Custom |
| `attention` | Attention | TritonBench | Custom |
| `conv2d` | Convolution | TritonBench | Custom |
| `layer_norm` | Normalization | TorchInductor ref. | Custom |
| `rms_norm` | Normalization | TritonBench | Custom |
| `add` | EW / Red. | TritonBench | Custom |
| `mul` | EW / Red. | TritonBench | Custom |
| `relu` | EW / Red. | TritonBench | Custom |
| `leaky_relu` | EW / Red. | TritonBench | Custom |
| `softmax` | EW / Red. | TritonBench | Custom |
| `log_softmax` | EW / Red. | TritonBench | Custom |
| `logsumexp` | EW / Red. | TritonBench | Custom |
| `swiglu` | EW / Red. | TritonBench | Custom |
| `argmax` | EW / Red. | TritonBench | Custom |
| `max_reduction` | EW / Red. | TritonBench | Custom |
| `mean_reduction` | EW / Red. | TritonBench | Custom |
| `cross_entropy` | EW / Red. | TritonBench | Custom |
| `matrix_transpose` | EW / Red. | TritonBench | Custom |
| `index_select` | EW / Red. | TritonBench | Custom |
| `embedding` | EW / Red. | TritonBench | Custom |

EW / Red. = element-wise or reduction. "Custom" = re-implementation following the same interface as TritonBench.

---

## Baseline Specifications

| Kernel category | Baseline |
|---|---|
| GEMM | cuBLAS via `torch.matmul` |
| Convolution | `nn.Conv2d` (NCHW, default cuDNN algorithm selection) |
| Normalization | `F.layer_norm` / `F.rms_norm` |
| Element-wise / Reduction | PyTorch eager |
| Attention | `F.scaled_dot_product_attention` with FlashAttention backend (`enable_flash_sdp(True)`) |

---

## Measurement Protocol

**End-to-end timing.** Host-synchronized GPU time via `time.perf_counter()` bracketed by
`torch.cuda.synchronize()`. Each workload uses 10 warm-up iterations followed by 100 timed
iterations; we report the median. Each workload runs in isolation.

**Clock locking.**

| GPU | Graphics lock | Memory lock |
|---|---|---|
| A100-SXM4-40GB | 1215 MHz | 1215 MHz |
| GH200 | 1320 MHz | 2619 MHz |

Run-to-run relative std-dev = 0.0–0.9% across 100-iteration windows on locked clocks.

**Nsight Compute counters.** Seven counters per kernel from a single `ncu` execution, verified
within 2% across five repeats: global-load efficiency, sectors per request, registers per thread,
register spills, long-scoreboard stall cycles, barrier stall cycles, L2 sector hit rate.

**KernelBench evaluator.** For the passes-but-slow demonstration we invoke KernelBench's
`eval_kernel_against_ref`, which overrides candidate `get_inputs`/`get_init_inputs` with the
reference's, checks output equivalence at the same per-dtype tolerances as our suite, and warns
only on >10× speedup (admitting any slowdown). See `experiments/results/NVIDIA_GH200_480GB/passes_but_slow.csv`.

---

## Hardware and Software

### Primary GPUs

| Property | A100-SXM4-40GB | GH200 |
|---|---|---|
| Architecture | Ampere (sm_80) | Grace Hopper (sm_90) |
| SMs | 108 | 132 |
| Memory | 40 GB HBM2e | 96 GB HBM3 |
| Peak mem. BW | ~1.5 TB/s | ~4.0 TB/s |
| Peak FP16 TC | ~312 TFLOP/s | ~989.5 TFLOP/s |
| L2 cache | 40 MB | 60 MB |
| TDP | 400 W | 900 W (SoC) |
| Clock lock (g/m) | 1215 / 1215 MHz | 1320 / 2619 MHz |
| CUDA Toolkit | 12.8 | 12.8 |

### Cross-Architecture Controls

| GPU | Role | Architecture |
|---|---|---|
| A100-PCIE-40GB | Form-factor control | sm_80 |
| H100-80GB-HBM3 | Cross-family control | sm_90 (Hopper) |

### Software Stack

| Package | Version |
|---|---|
| PyTorch | 2.8.0+cu128 |
| Triton | 3.4.0 |
| TileLang | 0.1.6.post1 |
| cuDNN | bundled with PyTorch |
| Nsight Compute | 2026.2.0 |

---

## Cross-Architecture Generalization

Median library efficiency (E_lib %) per operator category, untuned defaults.
The qualitative pattern (GEMM/element-wise competitive; convolution and TileLang normalization
severely behind) is preserved across all four GPUs.

| Category | Triton SXM4 | Triton PCIE | Triton H100 | TileLang SXM4 | TileLang PCIE | TileLang H100 |
|---|--:|--:|--:|--:|--:|--:|
| GEMM | 77.3% | 75.5% | 68.4% | 69.4% | 60.3% | 69.2% |
| Convolution | 31.6% | 38.0% | 37.7% | 9.5% | 11.1% | 7.8% |
| Normalization | 121.5% | 137.3% | 130.5% | 1.0% | 0.8% | 0.9% |
| Element-wise / Red. | 65.3% | 73.0% | 72.9% | 51.0% | 55.3% | 60.9% |

Full data and generator script: [`experiments/results/cross_arch/`](experiments/results/cross_arch/).

---

## Key Optimization Results

Five kernel-optimization campaigns (AKO4ALL). Each entry is a single-kernel iterative run
(1 iteration = 1 source edit + 1 benchmark run + correctness check + git commit).

| Kernel | DSL | Input | Before | After | Speedup | Iterations | Dominant pattern |
|---|---|---|--:|--:|--:|--:|---|
| layer_norm | TileLang | (8192, 8192) bf16 | 1090.0 ms | 0.89 ms | **1224×** | 18 | `T.serial → T.reduce` + native bf16 I/O |
| rms_norm | TileLang | (8192, 8192) fp16 | 716.0 ms | 0.90 ms | **796×** | 2 | `T.serial → T.reduce` + native fp16 I/O |
| argmax | TileLang | (8192, 32768) fp16 | 16.2 ms | 1.75 ms | **9.26×** | 13 | Tiled shared-mem loads + native fp16 I/O |
| matmul | Triton | (4096, 4096) fp16 | 2.71 ms | 1.63 ms | **1.66×** | 6 | `@triton.autotune` + `GROUP_SIZE_M` L2 swizzle |
| conv2d | Triton | (32, 256, 128, 128) fp16 | 32.1 ms | 12.5 ms | **2.57×** | 15 | Implicit GEMM + fp16 tensor cores + autotune |

Source: [`AKO4ALL/results/optimization_results.csv`](AKO4ALL/results/optimization_results.csv).
Per-iteration logs: `AKO4ALL/results/optimized/<kernel>_iterations.md`.

---

## Running the Artifact

```bash
pip install torch triton tilelang     # ncu (Nsight Compute) also needed for profiling

# Correctness
python ViperBench/run_all.py                          # all 22 kernels, prints summary
python ViperBench/<kernel>/test.py                    # single kernel

# Latency + peak-memory benchmarks → ViperBench/results/profile.<gpu>.csv
python ViperBench/benchmark.py                        # PyTorch + Triton
python ViperBench/benchmark_tilelang.py               # TileLang
python ViperBench/benchmark_tuned.py                  # all impls including tuned

# Auto-tuning sweep (writes results/tuning_cache.json)
cd ViperBench && python -m tuning.sweep --all

# Rebuttal experiments (outputs namespaced under experiments/results/<gpu_slug>/)
bash experiments/run_all.sh                           # serial master runner
python experiments/exp_fp32_gemm.py                   # FP32 TF32-lowering root cause
python experiments/exp_conv_filters.py                # conv filter sweep
python experiments/exp_significance.py                # clock-locked significance
bash experiments/ncu_counters.sh <kernel> <impl> <size>

# Cross-arch replay
# See experiments/A100_H100_RUNBOOK.md

# Paper build (from paper-latex-project/)
cd paper-latex-project && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

