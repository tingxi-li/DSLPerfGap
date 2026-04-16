# ViperBench — GPU Kernel Benchmark & Porting Project

## Project Overview
This project benchmarks GPU kernels across PyTorch (eager + compile), Triton, and TileLang implementations. Each kernel must produce **numerically identical outputs** across all implementations. The evaluation infrastructure measures correctness, latency, memory bandwidth, and peak memory usage.

## Directory Structure
```
ViperBench/
  kernels/                    # 240 consolidated kernels
    <kernel_name>/
      metadata.json           # Category, difficulty, provenance
      input_config.json       # Shape/dtype/param sweeps + tolerances
      reference.py            # PyTorch reference with reference(inputs) → outputs
      triton_impl.py          # Triton implementation (or NOT_IMPLEMENTED stub)
      tilelang_impl.py        # TileLang implementation (or NOT_IMPLEMENTED stub)
      input_gen.py            # (optional) Local override for input generation
      metrics.py              # (optional) Local override for FLOP/byte counting
  viperbench/                 # Evaluation infrastructure
    runner.py                 # Orchestration: CLI + main eval loop
    input_gen.py              # Central input generation with category dispatch
    validate.py               # Numerical correctness checking
    profile.py                # Latency timing + peak memory measurement
    metrics.py                # FLOP/byte counting, SOL computation
    analyze.py                # Results aggregation, tables, coverage
    utils.py                  # Dtype mapping, hardware detection, logging
    __main__.py               # python -m viperbench entry point
  configs/
    defaults.json             # Global profiling/correctness/runner defaults
    hardware/                 # GPU hardware specs
      rtx4000_ada.json
      rtx6000_ada.json
      a100_80gb.json
      h100_sxm.json
  scripts/
    smoke_test.py             # Quick eager+compile smoke test for all 240 kernels
  results/                    # JSON + CSV eval output (per-kernel subdirs)

kernel_samples/               # Consolidated source kernels (supersedes KernelBench_dedup/)
  activation/                 # Activation op sources
  normalization/              # Normalization op sources
  ...                         # One subdir per category

docs/
  evaluation-contruction-sepc.md  # Full evaluation infrastructure specification
```

## Kernel Interface

### Direct-style kernels (82 kernels: primitives + standalone ops)

```python
def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Takes named input tensors, returns named output tensors."""
    x = inputs["input"]
    return {"output": torch.relu(x)}
```

### Model-style kernels (158 kernels: fused ops + complete models)

```python
class Model(nn.Module):
    def __init__(self, ...): ...
    def forward(self, ...): ...

def get_inputs(): ...          # Returns list of CPU tensors
def get_init_inputs(): ...     # Returns constructor args
def get_test_inputs(): ...     # Returns CUDA-ready inputs
def run(*args): ...            # One-call entry point

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Wraps Model — takes inputs["input"], returns {"output": ...}"""
```

## Evaluation Runner

```bash
# All commands run from ViperBench/ directory
cd ViperBench

# List all 240 kernels and implementation status
python -m viperbench.runner --list

# Run one kernel (correctness + profiling)
python -m viperbench.runner --kernel matmul --sweep prioritized

# Correctness only (skip profiling)
python -m viperbench.runner --kernel matmul --correctness-only

# Specific config
python -m viperbench.runner --kernel matmul --shape prefill --dtype fp16

# Run all kernels
python -m viperbench.runner --all --sweep prioritized

# Specify hardware config (auto-detected if omitted)
python -m viperbench.runner --kernel matmul --hardware configs/hardware/a100_80gb.json
```

### Evaluation flow per (kernel, config, implementation):
1. **Load** — input_config.json, metadata.json, hardware spec, discover implementations
2. **Generate inputs** — seeded (torch.manual_seed(42)), category-dispatched or local override
3. **Reference run** — golden outputs from reference.py
4. **For each impl** — correctness check → profiling (latency + memory) → derived metrics (TFLOPS, SOL%)
5. **Write results** — `results/<kernel>/timing.json` + `results/summary.csv`

Implementations tested: `pytorch_eager`, `pytorch_compile`, `triton_impl`, `tilelang_impl`

## Smoke Test

```bash
# Quick pass/fail for all 240 kernels in eager + compile modes
python ViperBench/scripts/smoke_test.py
python ViperBench/scripts/smoke_test.py --mode eager
python ViperBench/scripts/smoke_test.py --mode compile
python ViperBench/scripts/smoke_test.py --kernel "fused_*"
```

Current status: **240/240 eager, 239/240 compile** (batch_norm compile fails due to Dynamo `Tensor.set_()` tracing limitation).

## Analysis

```bash
cd ViperBench
python -m viperbench.analyze --summary        # Per-kernel speedup table
python -m viperbench.analyze --coverage        # Implementation coverage matrix
python -m viperbench.analyze --speedups        # Geometric mean speedups vs eager
python -m viperbench.analyze --category matmul # Per-category breakdown
```

## Kernel Categories (22 categories, 240 kernels)

**Primitive Operations:**
matmul, conv, activation, normalization, pooling, reduction, loss, attention,
cumulative, embedding, dropout, quantization, elementwise, specialized

**Fused Operations:**
fused_conv, fused_convtranspose, fused_gemm, fused_matmul

**Complete Models:**
model_cnn, model_transformer, model_rnn, model_other

## Numerical Correctness Criteria

| dtype   | atol   | rtol   |
|---------|--------|--------|
| float32 | 1e-5   | 1e-5   |
| float16 | 1e-2   | 1e-2   |
| bfloat16| 1e-2   | 1e-2   |

Per-kernel tolerances in `input_config.json` override these defaults.
Integer outputs (argmax, argmin, embedding) use exact match.

## Hardware Configs

| GPU | File | Memory BW | FP16 TFLOPS | SMs |
|-----|------|-----------|-------------|-----|
| RTX 4000 Ada | `rtx4000_ada.json` | 272 GB/s | 38.4 | 48 |
| RTX 6000 Ada | `rtx6000_ada.json` | 960 GB/s | 182.2 | 142 |
| A100 SXM4 80GB | `a100_80gb.json` | 2039 GB/s | 312 | 108 |
| H100 SXM5 80GB | `h100_sxm.json` | 3350 GB/s | 1979 | 132 |

Hardware is auto-detected from `torch.cuda.get_device_name()` and matched against configs.

## TileLang Reference

### Installation
```bash
pip install tilelang
```

### Key Primitives
```python
import tilelang
import tilelang.language as T

@tilelang.jit
def my_kernel(M, N, ...):
    @T.prim_func
    def kernel_body(A: T.Tensor((M, N), "float16"), ...):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            C_local  = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel_body
```

### Useful Constructs
- `T.Kernel(grid_x, grid_y, threads=N)` — launch config
- `T.alloc_shared(shape, dtype)` — shared memory
- `T.alloc_fragment(shape, dtype)` — register fragment
- `T.copy(src, dst)` — async tile copy
- `T.gemm(A, B, C)` — tile GEMM
- `T.reduce(src, dst, op)` — tile reduction
- `T.clear(buf)` — zero a buffer
- `T.Pipelined(N, num_stages=k)` — software pipelined loop
- `T.ceildiv(a, b)` — ceiling division
- `T.use_swizzle(panel_size=10)` — L2 cache locality hint

## Kernel Porting Workflow

### Per-Kernel Process
1. Pick a kernel from `ViperBench/kernels/`
2. Read `reference.py` to understand the algorithm, shapes, and dtypes
3. Read `input_config.json` for shape variants and tolerances
4. Write `tilelang_impl.py` (or `triton_impl.py`) in the same directory
5. Test: `cd ViperBench && python -m viperbench.runner --kernel <name> --correctness-only`
6. If test fails, read the full error, fix the impl, retry (up to **20 attempts**)
7. On pass: run with profiling to get latency/SOL metrics

### Retry Strategy
When a TileLang kernel fails:
1. Print the full error/traceback
2. Diagnose: shape mismatch? wrong indexing? wrong dtype? missing `T.clear()`?
3. Edit `tilelang_impl.py` with a targeted fix
4. Re-run test immediately
5. Repeat up to 20 times; if still failing after 20, write a `FAILED` entry and move on

## Adding a New Kernel

Standard kernel (uses central input_gen + metrics):
```
1. mkdir ViperBench/kernels/<name>
2. Write metadata.json (category, difficulty, provenance)
3. Write input_config.json (shape/dtype sweeps, tolerances)
4. Write reference.py with reference(inputs) -> outputs
5. Create triton_impl.py stub (NOT_IMPLEMENTED = True)
6. Create tilelang_impl.py stub (NOT_IMPLEMENTED = True)
7. Test: python -m viperbench.runner --kernel <name> --correctness-only
```

Nonstandard kernel (needs custom input/metrics):
```
1-6. Same as above
7. Write kernels/<name>/input_gen.py with generate(config, dtype) -> tensors
8. Write kernels/<name>/metrics.py with compute_flops(config) and compute_bytes(config, dtype)
9. Test: cd ViperBench && python -m viperbench.runner --kernel <name> --correctness-only
```

## Autonomy Rules
- **Always allowed**: read/write files in `./`, run `python`, `grep`, `nvidia-smi`, `pip install`
- **Blocked**: `sudo`, network calls outside pip installs
- Work sequentially through kernels; do not skip ahead unless a kernel exceeds 20 retries

## Common TileLang Pitfalls
- Always call `T.clear(C_local)` before accumulation
- Block tile sizes must divide evenly into tensor dims, or use padding
- `T.copy` direction: `T.copy(global_src[offset], shared_dst)` — order matters
- `T.gemm` expects (A, B, C) where C is the accumulator (fragment, not shared)
- For elementwise ops, iterate with `for i, j in T.Parallel(M, N):` or use thread indices
- `threads=` in `T.Kernel` must be a multiple of 32 (warp size)
- Output `T.copy` from fragment to global: `T.copy(C_local, C[by*bM, bx*bN])`

## Auto-Compact Rules
When context is compacted, the summary MUST preserve:
- **Current kernel name** and which step of the porting workflow you are on
- **Retry count** for the current kernel (e.g., "attempt 5/20")
- **Last error message/traceback** if currently debugging a failing kernel
- **List of completed kernels** and their PASS/FAIL status
- **Any non-obvious findings** about TileLang behavior discovered during porting

The summary may discard:
- Full file contents already written to disk (re-read as needed)
- Intermediate debugging attempts that were superseded by later fixes
- Verbose tool output from successful operations
- Redundant shape/dtype details that can be re-read from source files
