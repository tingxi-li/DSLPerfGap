# ViperBench

A GPU kernel benchmarking suite comparing **PyTorch**, **Triton**, and **TileLang** implementations across 22 common deep learning operations, with an automated optimization framework ([AKO4ALL](#ako4all--automated-kernel-optimization)) for closing performance gaps.

## Kernels

| Category | Kernels |
|----------|---------|
| Elementwise | `add`, `mul`, `relu`, `leaky_relu`, `swiglu` |
| Reduction | `argmax`, `max_reduction`, `mean_reduction`, `softmax`, `log_softmax`, `logsumexp` |
| Normalization | `layer_norm`, `rms_norm` |
| Linear algebra | `matmul`, `batched_matmul`, `linear_activation`, `conv2d` |
| Memory | `embedding`, `index_select`, `matrix_transpose` |
| Other | `attention`, `cross_entropy` |

Each kernel has three implementations under `ViperBench/<kernel>/`:
- `pytorch_impl.py` — reference (torch built-ins / cuDNN)
- `triton_impl.py` — custom Triton kernel
- `tilelang_impl.py` — custom TileLang kernel

---

## Quick Start

### Requirements

```bash
pip install torch triton tilelang
```

NVIDIA GPU with CUDA required. `ncu` (Nsight Compute) is needed only for AKO4ALL profiling.

### Run correctness tests

```bash
# Test all kernels
python ViperBench/run_all.py

# Test a single kernel
python ViperBench/layer_norm/test.py
```

### Benchmark latency and memory

```bash
# PyTorch + Triton baselines
python ViperBench/benchmark.py

# TileLang implementations
python ViperBench/benchmark_tilelang.py

# All implementations including tuned variants
python ViperBench/benchmark_tuned.py
```

Results are written to `ViperBench/results/profile.csv`:

```
kernel, size, impl, input_desc, latency_ms, peak_memory_mb
layer_norm, large, pytorch, "x:(8192,8192) bf16", 0.87, ...
layer_norm, large, triton,  "x:(8192,8192) bf16", 0.92, ...
layer_norm, large, tilelang,"x:(8192,8192) bf16", 0.89, ...
```

`impl` values: `pytorch`, `triton`, `triton_tuned`, `tilelang`, `tilelang_tuned`

---

## Tuning

Each kernel loads tuned hyperparameters (block sizes, thread counts, etc.) from a shared cache at startup:

```
ViperBench/results/tuning_cache.json   # per-kernel, per-GPU-arch configs
ViperBench/tuning/                     # cache loader + sweep utilities
```

To run a tuning sweep for a specific kernel, see `ViperBench/tuning/sweep.py`.

---

## AKO4ALL — Automated Kernel Optimization

`AKO4ALL/` is an agentic optimization loop that iteratively rewrites a kernel for maximum performance using Claude Code as the optimizer.

### Prepare a kernel for optimization

```bash
cd AKO4ALL
python prepare_kernel.py <kernel> <triton|tilelang>

# Examples:
python prepare_kernel.py layer_norm tilelang
python prepare_kernel.py matmul triton
```

This generates:
- `input/reference.py` — PyTorch golden reference in KernelBench format
- `solution/kernel.py` — DSL kernel to optimize
- `scripts/bench.sh` — benchmark + trajectory tracking script

### Verify baseline

```bash
bash scripts/bench.sh baseline
# COMPILED: True
# CORRECT:  True
# RUNTIME:  0.89
# SPEEDUP:  0.98x
```

### Run the optimization agent

```bash
claude --dangerously-skip-permissions
# Then: "Follow the instructions in TASK.md. Optimize for up to 20 iterations."
```

The agent profiles with `ncu`, iterates (modify → bench → commit), and logs every attempt to `ITERATIONS.md`. All solutions are snapshotted under `scripts/trajectory/`.

### Rules (see `AKO4ALL/HINTS.md`)

- TileLang kernels must stay in TileLang; Triton must stay in Triton — no language switching
- Profile before optimizing; re-profile after 3 consecutive non-improvements
- Do not install packages

### Optimization results

`AKO4ALL/results/optimization_results.csv` records completed campaigns:

| kernel | before | after | speedup |
|--------|--------|-------|---------|
| layer_norm (TileLang) | 1090 ms | 0.89 ms | 1224× |
| rms_norm (TileLang) | 716 ms | 0.90 ms | 796× |
| argmax (TileLang) | 16.2 ms | 1.75 ms | 9× |
| matmul (Triton) | 2.71 ms | 1.63 ms | 1.66× |
| conv2d (Triton) | 32.1 ms | 12.5 ms | 2.57× |

---

## Repository Layout

```
ViperBench/
├── <kernel>/
│   ├── pytorch_impl.py
│   ├── triton_impl.py
│   ├── tilelang_impl.py
│   └── test.py
├── benchmark.py            # PyTorch + Triton profiling
├── benchmark_tilelang.py   # TileLang profiling
├── benchmark_tuned.py      # Tuned variant profiling
├── run_all.py              # Correctness test runner
├── tuning/                 # Tuning cache + sweep tools
└── results/
    ├── profile.csv         # All benchmark results
    └── slow_kernels.csv    # Kernels slower than PyTorch (large inputs, tuned only)

AKO4ALL/
├── prepare_kernel.py       # Set up optimization environment for any ViperBench kernel
├── TASK.md                 # Agent optimization protocol
├── HINTS.md                # Optimization constraints
├── bench/kernelbench/      # Self-contained evaluator (correctness + timing)
├── context/                # Reference docs: GPU analysis, known issues, tuning guides
├── solution/               # Active kernel being optimized
└── results/optimized/      # Saved best solutions + iteration logs
```
