# ViperBench Evaluation Infrastructure Specification

## 1. Overview

This document specifies the evaluation infrastructure for ViperBench. The
infrastructure measures four things for every (kernel, config, implementation)
tuple:

1. **Numerical correctness** — does the output match the reference?
2. **End-to-end latency** — how long does one kernel invocation take?
3. **Memory bandwidth (achieved)** — what fraction of peak bandwidth is used?
4. **Peak memory usage** — how much GPU memory does the implementation consume?

The infrastructure is split into **universal modules** (in `viperbench/`)
that work for all kernels, and **per-kernel files** (in `kernels/<name>/`)
that define kernel-specific behavior.

---

## 2. Module Responsibilities

```
viperbench/
├── runner.py          # Orchestration: the main evaluation loop
├── input_gen.py       # Central input generation with category dispatch
├── validate.py        # Numerical correctness checking
├── profile.py         # Latency timing + peak memory measurement
├── metrics.py         # FLOP/byte counting with category dispatch
├── analyze.py         # Results aggregation, tables, plots
└── utils.py           # Shared helpers (dtype mapping, device setup, logging)
```

| Module | Scope | Kernel-specific? |
|---|---|---|
| `runner.py` | Universal | No — same loop for all kernels |
| `input_gen.py` | Central + override | Category dispatch; per-kernel override if needed |
| `validate.py` | Universal | No — same comparison logic for all |
| `profile.py` | Universal | No — same timing/memory protocol for all |
| `metrics.py` | Central + override | Category dispatch; per-kernel override if needed |
| `analyze.py` | Universal | No — reads results uniformly |
| `utils.py` | Universal | No |

**Override mechanism:** If a kernel directory contains a local `input_gen.py`
or `metrics.py`, the runner uses that instead of the central version. This
handles nonstandard kernels (jagged ops, Mamba, custom fusions) without
polluting the central code.

```
kernels/matmul/              # Uses central input_gen + central metrics
    metadata.json
    input_config.json
    reference.py
    triton_impl.py
    tilelang_impl.py

kernels/jagged_softmax/      # Uses LOCAL input_gen + LOCAL metrics
    metadata.json
    input_config.json
    reference.py
    triton_impl.py
    tilelang_impl.py
    input_gen.py              # Override: custom jagged tensor generation
    metrics.py                # Override: custom byte counting for jagged
```

---

## 3. `runner.py` — Orchestration

### 3.1 CLI Interface

```bash
# Run one kernel, priority sweep
python -m viperbench.runner --kernel matmul --sweep prioritized

# Run one kernel, full sweep
python -m viperbench.runner --kernel matmul --sweep full

# Run all kernels
python -m viperbench.runner --all --sweep prioritized

# Run specific config
python -m viperbench.runner --kernel matmul \
    --shape prefill --dtype fp16 --transpose NN --structure dense

# Correctness only (skip profiling)
python -m viperbench.runner --kernel matmul --correctness-only

# List available kernels and their implementation status
python -m viperbench.runner --list

# Specify hardware config (auto-detected if omitted)
python -m viperbench.runner --kernel matmul --hardware configs/hardware/a100_80gb.json
```

### 3.2 Execution Flow

For each (kernel, config) pair:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LOAD                                                     │
│    Read input_config.json                                   │
│    Read metadata.json                                       │
│    Load hardware spec (auto-detect or --hardware flag)      │
│    Resolve input_gen: local override or central             │
│    Resolve metrics: local override or central               │
│    Discover implementations: reference, triton, tilelang    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ 2. GENERATE INPUTS                                          │
│    Call input_gen.generate(config) → dict[str, Tensor]      │
│    Seed: torch.manual_seed(42), torch.cuda.manual_seed(42) │
│    All tensors on CUDA, in requested dtype                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ 3. REFERENCE RUN                                            │
│    Run reference.py at fp32 → golden_outputs                │
│    This is the correctness ground truth                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ 4. FOR EACH IMPLEMENTATION                                  │
│    [pytorch_eager, pytorch_compile, triton_impl, tilelang]  │
│                                                             │
│    4a. Check NOT_IMPLEMENTED flag → skip if True            │
│                                                             │
│    4b. CORRECTNESS (validate.py)                            │
│        Run impl once → test_outputs                         │
│        Compare test_outputs vs golden_outputs               │
│        Record: pass/fail, max_abs_err, max_rel_err          │
│        If fail → skip profiling, log error details          │
│                                                             │
│    4c. PROFILING (profile.py) — skip if --correctness-only  │
│        Measure latency: warmup + timed iterations           │
│        Measure peak memory: reset → run → read              │
│        Record: median_us, mean_us, min_us, max_us, std_us  │
│        Record: peak_memory_allocated_mb,                    │
│                peak_memory_reserved_mb                      │
│                                                             │
│    4d. METRICS (metrics.py)                                 │
│        Compute theoretical FLOPs for this config            │
│        Compute theoretical bytes for this config            │
│        Derive: achieved_tflops, achieved_bw_gb_s, sol_pct   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ 5. WRITE RESULTS                                            │
│    Write timing.json, correctness.json to results/          │
│    Append to summary.csv                                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Implementation Discovery

The runner scans each kernel directory for implementation files:

```python
IMPL_PATTERNS = {
    "pytorch_eager":   "reference.py",    # run with torch.no_grad(), no compile
    "pytorch_compile": "reference.py",    # wrapped in torch.compile()
    "triton_impl":     "triton_impl.py",
    "tilelang_impl":   "tilelang_impl.py",
}
```

For each file, the runner:
1. Imports the module.
2. Checks for `NOT_IMPLEMENTED = True` → skip.
3. Extracts the callable: `reference()` from reference.py, `kernel()` from
   DSL impl files.
4. For `pytorch_compile`: wraps the reference function in
   `torch.compile(fn, mode="reduce-overhead")` and runs a throwaway call
   to trigger compilation before profiling.

Additional DSL variants (`triton_naive.py`, `triton_autotuned.py`) are
discovered by globbing `*_impl.py` and `*_naive.py` / `*_autotuned.py`.

### 3.4 Sweep Configuration

The runner reads `sweep_mode` from `input_config.json` and generates the
list of configs to evaluate:

- `"prioritized"`: Uses `priority_sweep` to select axis subsets. Axes set to
  `"all"` include every value; axes set to a list include only those values.
  Computes Cartesian product of selected values.
- `"full"`: Cartesian product of all values in all axes.
- `"custom"`: Reads explicit list of config dicts from `custom_sweep`.

The runner logs the total config count before starting and shows progress.

---

## 4. `input_gen.py` — Input Generation

### 4.1 Interface

```python
def generate(
    kernel_name: str,
    category: str,
    config: dict,
    dtype: str,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Generate input tensors for a kernel.

    Args:
        kernel_name: e.g., "matmul"
        category: from metadata.json, e.g., "matmul", "conv", "attention"
        config: single config dict from input_config.json params,
                e.g., {"name": "prefill", "M": 2048, "N": 4096, "K": 4096,
                       "batch": 1, "transpose": "NN", "structure": "dense"}
        dtype: e.g., "fp16"
        device: "cuda" or "cpu"
        seed: random seed for reproducibility

    Returns:
        dict of named tensors, e.g., {"A": tensor, "B": tensor}
    """
```

### 4.2 Category Dispatch

The central `input_gen.py` dispatches to category-specific generators:

```python
GENERATORS = {
    "matmul":             _gen_matmul,
    "conv":               _gen_conv,
    "attention":          _gen_attention,
    "normalization":      _gen_normalization,
    "activation":         _gen_activation,
    "elementwise":        _gen_elementwise,
    "reduction":          _gen_reduction,
    "loss":               _gen_loss,
    "cumulative":         _gen_cumulative,
    "pooling":            _gen_pooling,
    "embedding":          _gen_embedding,
    "positional_encoding":_gen_rope,
    "dropout":            _gen_activation,    # same as activation
    "fused_matmul":       _gen_fused_matmul,
    "fused_gemm":         _gen_fused_gemm,
    "fused_conv":         _gen_fused_conv,
    "fused_convtranspose":_gen_fused_conv,    # same logic, swap C_in/C_out
    "model_cnn":          _gen_model,
    "model_transformer":  _gen_model,
    "model_rnn":          _gen_model,
    "model_other":        _gen_model,
    "specialized":        None,               # must use local override
}
```

If `GENERATORS[category]` is `None`, the runner looks for a local
`input_gen.py` in the kernel directory. If neither exists, the runner
raises an error.

### 4.3 Override Resolution

```python
def resolve_input_gen(kernel_dir: Path, category: str):
    """
    Returns the generate() function to use.
    Local override takes priority over central dispatch.
    """
    local_gen = kernel_dir / "input_gen.py"
    if local_gen.exists():
        module = import_module_from_path(local_gen)
        return module.generate
    
    if category in GENERATORS and GENERATORS[category] is not None:
        return lambda config, dtype, **kw: GENERATORS[category](config, dtype, **kw)
    
    raise ValueError(
        f"No input generator for category '{category}'. "
        f"Add a local input_gen.py to kernels/{kernel_dir.name}/"
    )
```

### 4.4 Category Generator Specifications

Each generator must accept a config dict (from `input_config.json` params)
and a dtype string, and return a dict of named tensors.

#### Matmul

```
Input config keys:  M, N, K, batch, transpose, structure
Output tensors:     {"A": (batch?, M, K), "B": (batch?, K, N)}

transpose handling:
  "NN" → A is (M, K), B is (K, N)
  "NT" → A is (M, K), B is (N, K)  [caller expects impl to transpose B]
  "TN" → A is (K, M), B is (K, N)  [caller expects impl to transpose A]

structure handling:
  "dense"            → torch.randn
  "diagonal"         → torch.diag(torch.randn(min(M,K)))
  "upper_triangular" → torch.triu(torch.randn(M, K))
  "lower_triangular" → torch.tril(torch.randn(M, K))
  "symmetric"        → A = randn(M,M); A = (A + A.T) / 2 (requires M==K)
  "sparse_X"         → torch.randn * (torch.rand > X/100)

batch handling:
  batch=1  → omit batch dim (2D tensors)
  batch>1  → include batch dim (3D tensors)
```

#### Conv

```
Input config keys:  batch, C_in, H, W, C_out, kernel_size, stride,
                    padding, dilation, groups
Output tensors:     {"input": (B, C_in, H, W),
                     "weight": (C_out, C_in/groups, kH, kW),
                     "bias": (C_out,) or None}

For conv1d: {"input": (B, C_in, L), "weight": (C_out, C_in/groups, K)}
For conv3d: {"input": (B, C_in, D, H, W), "weight": (C_out, C_in/groups, kD, kH, kW)}
```

#### Attention

```
Input config keys:  batch, heads, seq_q, seq_kv, head_dim, kv_heads, mask
Output tensors:     {"Q": (B, H, Sq, D),
                     "K": (B, KVH, Skv, D),
                     "V": (B, KVH, Skv, D),
                     "mask": (Sq, Skv) or None}

mask handling:
  "none"     → mask=None
  "causal"   → lower-triangular boolean mask
  "sparse_X" → random mask with X% entries masked
```

#### Normalization

```
Input config keys:  batch, dims, norm_type
Output tensors:     {"input": (B, *dims),
                     "weight": (normalized_shape,),
                     "bias": (normalized_shape,)}

norm_type determines normalized_shape:
  "layernorm"/"rmsnorm"     → last dim of dims
  "batchnorm"/"instancenorm" → first dim of dims (channels)
  "groupnorm"               → first dim, plus num_groups param
```

#### Activation / Elementwise / Dropout

```
Input config keys:  dims (list of ints)
Output tensors:     {"input": (*dims,)}

For elementwise with broadcasting:
  Additional keys: broadcast_pattern
  Output tensors:  {"A": (*dims,), "B": (*broadcast_dims,)}
```

#### Reduction

```
Input config keys:  dims (list of ints), reduce_dim
Output tensors:     {"input": (*dims,)}
Config also passes: reduce_dim (int) to the kernel
```

#### Loss

```
Input config keys:  batch, seq_len (nullable), num_classes
Output tensors:     {"prediction": (B, [S,] C),
                     "target": (B, [S])}

target generation:
  Classification → torch.randint(0, num_classes, (B, [S]))
  Regression     → torch.randn(same shape as prediction)
```

#### Model-Level

```
Input config keys:  batch, seq_len (for transformers/RNNs)
Output tensors:     Model-specific, determined by category:
  CNN:         {"input": (B, 3, 224, 224)}
  Transformer: {"input_ids": (B, S), "attention_mask": (B, S)}
  RNN:         {"input": (B, S, input_size), "h0": (layers, B, hidden)}
```

### 4.5 Local Override Interface

A per-kernel `input_gen.py` override must define:

```python
def generate(
    config: dict,
    dtype: str,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Same interface as central generate(), minus kernel_name/category."""
```

---

## 5. `validate.py` — Correctness Checking

### 5.1 Interface

```python
def check_correctness(
    reference_outputs: dict[str, torch.Tensor],
    test_outputs: dict[str, torch.Tensor],
    atol: float,
    rtol: float,
) -> ValidationResult:
    """
    Compare test outputs against reference.

    Returns:
        ValidationResult with fields:
            passed: bool
            per_tensor: dict[str, TensorComparison]
            error_message: str or None
    """

@dataclass
class TensorComparison:
    name: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    num_mismatched: int        # elements exceeding tolerance
    total_elements: int
    mismatch_fraction: float   # num_mismatched / total_elements
    shape_match: bool
    dtype_match: bool

@dataclass
class ValidationResult:
    passed: bool
    per_tensor: dict[str, TensorComparison]
    error_message: str | None
```

### 5.2 Comparison Rules

1. **Shape check:** If test tensor shape != reference tensor shape → fail
   immediately, report shape mismatch.

2. **Output key check:** If test output is missing keys that reference has,
   or has extra keys → fail.

3. **NaN/Inf check:** If test output contains NaN or Inf → fail, report
   count and positions.

4. **Numerical comparison:** Both tensors are cast to fp32, then compared
   element-wise:
   ```
   abs_diff = |test - ref|
   rel_diff = abs_diff / max(|ref|, atol)
   passed = all(abs_diff <= atol + rtol * |ref|)
   ```

5. **Integer output handling:** For kernels that return indices (argmax,
   argmin, embedding lookup), use exact match (`torch.equal`) instead of
   `allclose`. Detect this from output dtype: if int → exact match.

6. **Tolerance source:** Read from `input_config.json` `correctness` field,
   keyed by dtype:
   ```json
   "correctness": {
       "atol": {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-5},
       "rtol": {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-5},
       "reference_dtype": "fp32"
   }
   ```

### 5.3 Error Reporting

When validation fails, the result includes:
- Which tensor(s) failed
- Max absolute and relative errors
- Number and fraction of mismatched elements
- A sample of mismatched positions (first 5) with ref vs test values

This helps debug whether a failure is a catastrophic bug (all outputs wrong)
or a marginal precision issue (a few elements slightly off).

---

## 6. `profile.py` — Latency and Memory Measurement

### 6.1 Interface

```python
def profile_latency(
    fn: Callable,
    inputs: dict[str, torch.Tensor],
    warmup_iters: int = 10,
    timed_iters: int = 100,
    clear_l2: bool = True,
) -> LatencyResult:
    """
    Measure kernel execution latency.

    Returns:
        LatencyResult with fields:
            median_us: float
            mean_us: float
            min_us: float
            max_us: float
            std_us: float
            all_times_us: list[float]
    """

def profile_memory(
    fn: Callable,
    inputs: dict[str, torch.Tensor],
) -> MemoryResult:
    """
    Measure peak GPU memory consumption of a single kernel invocation.

    Returns:
        MemoryResult with fields:
            peak_allocated_mb: float    # actual tensor memory
            peak_reserved_mb: float     # allocator reserved (includes fragmentation)
            input_memory_mb: float      # memory of input tensors alone
            overhead_mb: float          # peak_allocated - input_memory (workspace + outputs)
    """

@dataclass
class LatencyResult:
    median_us: float
    mean_us: float
    min_us: float
    max_us: float
    std_us: float
    all_times_us: list[float]

@dataclass
class MemoryResult:
    peak_allocated_mb: float
    peak_reserved_mb: float
    input_memory_mb: float
    overhead_mb: float
```

### 6.2 Latency Measurement Protocol

```
1. SETUP
   - Verify GPU clocks are locked (warn if not)
   - Move all inputs to CUDA, ensure contiguous

2. WARMUP
   - Run fn(inputs) for warmup_iters iterations
   - Purpose: JIT compilation, CUDA context init, cache warming
   - For torch.compile: this is where compilation happens
   - Discard all timing data from warmup

3. TIMED ITERATIONS
   For i in range(timed_iters):
       a. [Optional] Clear L2 cache
          - Allocate a scratch buffer ≥ L2 size (e.g., 48 MB for A100)
          - Write to it to evict cached data
          - Free the buffer
       b. torch.cuda.synchronize()
       c. start_event = torch.cuda.Event(enable_timing=True)
          end_event = torch.cuda.Event(enable_timing=True)
       d. start_event.record()
       e. fn(inputs)
       f. end_event.record()
       g. torch.cuda.synchronize()
       h. elapsed_ms = start_event.elapsed_time(end_event)
       i. Record elapsed_ms * 1000 → elapsed_us

4. REPORT
   - Sort all_times_us
   - median = all_times_us[len // 2]
   - Report {median, mean, min, max, std, all_times_us}
```

**Why median over mean:** Outliers from OS interrupts, thermal throttling,
or GC pauses inflate the mean. Median is robust. Min is useful as a
theoretical "best possible" but is less reproducible.

**L2 cache clearing:** Enabled by default. Measures cold-cache performance,
which is the conservative estimate. Can be disabled to measure warm-cache
(best-case repeated invocation).

**Clock locking:** The profiler checks `nvidia-smi -q -d CLOCK` and warns
if GPU clocks are not locked. Provides the locking command:
`nvidia-smi -lgc <base_clock>,<base_clock>`.

### 6.3 Memory Measurement Protocol

```
1. torch.cuda.reset_peak_memory_stats()
2. torch.cuda.synchronize()

3. Compute input_memory_mb:
   sum(t.element_size() * t.nelement() for t in inputs.values()) / 1e6

4. fn(inputs)     # single invocation

5. torch.cuda.synchronize()

6. peak_allocated_mb = torch.cuda.max_memory_allocated() / 1e6
7. peak_reserved_mb = torch.cuda.max_memory_reserved() / 1e6
8. overhead_mb = peak_allocated_mb - input_memory_mb
```

**What overhead_mb captures:**
- Output tensor allocations
- Intermediate/workspace buffers (e.g., Triton's scratch space)
- Temporary tensors inside the kernel
- The difference between `allocated` and `reserved` shows PyTorch
  allocator fragmentation

**Run separately from latency:** Memory profiling is done in a single
invocation, not during the timed loop. `max_memory_allocated` is a
high-water mark that would accumulate across iterations.

### 6.4 torch.compile Special Handling

`torch.compile` has unique profiling requirements:

```
1. Compile phase (during warmup):
   compiled_fn = torch.compile(reference_fn, mode="reduce-overhead")
   compiled_fn(inputs)   # triggers compilation, slow
   compiled_fn(inputs)   # second call, may still be compiling
   compiled_fn(inputs)   # by now, should be compiled

2. Timing phase:
   Use compiled_fn in the standard latency loop.
   The compilation cost is NOT included in the reported latency.
   
3. Optional: record compilation time separately
   compile_start = time.perf_counter()
   compiled_fn(inputs)
   torch.cuda.synchronize()
   compile_end = time.perf_counter()
   compile_time_s = compile_end - compile_start
```

Report `compile_time_s` in results but do not include it in `median_us`.
Users care about steady-state throughput, not first-call compilation.

---

## 7. `metrics.py` — Theoretical FLOP and Byte Counting

### 7.1 Interface

```python
def compute_flops(
    category: str,
    config: dict,
) -> int | None:
    """
    Return theoretical FLOP count for this (category, config).
    Returns None if FLOP counting is not meaningful (e.g., embedding lookup).
    """

def compute_bytes(
    category: str,
    config: dict,
    dtype: str,
) -> int | None:
    """
    Return theoretical bytes transferred (read + write) for this config.
    Returns None if byte counting is not meaningful.
    """

def compute_arithmetic_intensity(
    category: str,
    config: dict,
    dtype: str,
) -> float | None:
    """FLOPs / bytes. Determines compute-bound vs memory-bound regime."""

def compute_sol(
    latency_us: float,
    flops: int | None,
    bytes_transferred: int | None,
    hardware: dict,
    dtype: str,
) -> SOLResult:
    """
    Compute Speed-of-Light metrics.

    Returns:
        SOLResult with fields:
            achieved_tflops: float or None
            sol_compute_pct: float or None    # achieved / peak TFLOPS
            achieved_bw_gb_s: float or None
            sol_memory_pct: float or None     # achieved / peak BW
            bottleneck: str                   # "compute" or "memory"
    """
```

### 7.2 Category FLOP Formulas

| Category | FLOP formula | Notes |
|---|---|---|
| matmul | `2 * M * N * K * batch` | Standard GEMM FLOPs |
| conv2d | `2 * B * C_out * H_out * W_out * C_in/groups * kH * kW` | Per output element |
| attention | `4 * B * H * Sq * Skv * D` | QK^T + softmax + AV |
| normalization | `~5 * numel` (mean, var, normalize, scale, shift) | Approximate |
| activation | `numel * ops_per_element` | ReLU=1, GELU≈8, sigmoid≈4 |
| reduction | `numel` | One op per element |
| loss | `numel * ops_per_element + reduction` | Varies by loss type |

For fused ops: sum of component FLOPs.

### 7.3 Category Byte Formulas

| Category | Bytes formula | Notes |
|---|---|---|
| matmul | `(M*K + K*N + M*N) * bytes_per_elem * batch` | Read A, B; write C |
| activation | `2 * numel * bytes_per_elem` | Read input, write output |
| normalization | `3 * numel * bytes_per_elem` | Read input+params, write output |
| reduction | `(numel + output_numel) * bytes_per_elem` | Read input, write reduced |

`bytes_per_elem`: fp16/bf16 = 2, fp32 = 4, fp64 = 8, int8 = 1.

### 7.4 SOL Computation

```python
# Achieved TFLOPS
achieved_tflops = flops / (latency_us * 1e-6) / 1e12

# SOL compute percentage
peak_tflops = hardware[f"peak_{dtype}_tflops"]
sol_compute_pct = achieved_tflops / peak_tflops * 100

# Achieved bandwidth
achieved_bw_gb_s = bytes_transferred / (latency_us * 1e-6) / 1e9

# SOL memory percentage
sol_memory_pct = achieved_bw_gb_s / hardware["memory_bandwidth_gb_s"] * 100

# Bottleneck classification
arithmetic_intensity = flops / bytes_transferred  # FLOPs per byte
ridge_point = peak_tflops * 1e12 / (hardware["memory_bandwidth_gb_s"] * 1e9)
if arithmetic_intensity > ridge_point:
    bottleneck = "compute"
else:
    bottleneck = "memory"
```

### 7.5 Override Resolution

Same pattern as `input_gen.py`:

```python
def resolve_metrics(kernel_dir: Path, category: str):
    local_metrics = kernel_dir / "metrics.py"
    if local_metrics.exists():
        module = import_module_from_path(local_metrics)
        return module.compute_flops, module.compute_bytes
    return central_compute_flops, central_compute_bytes
```

Local `metrics.py` must define `compute_flops(config) -> int` and
`compute_bytes(config, dtype) -> int`.

---

## 8. Results Format

### 8.1 Per-Config Result Record

Each (kernel, config, implementation) evaluation produces:

```python
@dataclass
class EvalResult:
    # Identity
    kernel: str
    config_name: str
    config: dict
    implementation: str
    
    # Correctness
    correct: bool
    max_abs_error: float | None
    max_rel_error: float | None
    mismatch_fraction: float | None
    
    # Latency
    median_us: float | None       # None if correctness failed
    mean_us: float | None
    min_us: float | None
    max_us: float | None
    std_us: float | None
    
    # Memory
    peak_allocated_mb: float | None
    peak_reserved_mb: float | None
    overhead_mb: float | None
    
    # Derived metrics
    achieved_tflops: float | None
    sol_compute_pct: float | None
    achieved_bw_gb_s: float | None
    sol_memory_pct: float | None
    bottleneck: str | None        # "compute" or "memory"
    
    # Metadata
    hardware: str
    dtype: str
    timestamp: str
```

### 8.2 `results/timing.json`

Full results with metadata for reproducibility:

```json
{
  "kernel": "matmul",
  "hardware": "NVIDIA A100-SXM4-80GB",
  "cuda_version": "12.4",
  "pytorch_version": "2.4.0",
  "triton_version": "3.0.0",
  "tilelang_version": "0.1.6",
  "gpu_clock_mhz": 1410,
  "timestamp": "2026-04-09T12:00:00Z",
  "results": [
    {
      "config": {
        "shape": "prefill", "M": 2048, "N": 4096, "K": 4096,
        "dtype": "fp16", "transpose": "NN", "structure": "dense"
      },
      "implementations": {
        "pytorch_eager": {
          "correct": true,
          "max_abs_error": 0.0,
          "median_us": 245.3,
          "mean_us": 248.1,
          "min_us": 243.0,
          "max_us": 312.5,
          "std_us": 7.2,
          "peak_allocated_mb": 128.0,
          "peak_reserved_mb": 256.0,
          "overhead_mb": 64.0,
          "achieved_tflops": 278.4,
          "sol_compute_pct": 89.2,
          "achieved_bw_gb_s": null,
          "sol_memory_pct": null,
          "bottleneck": "compute"
        },
        "pytorch_compile": { ... },
        "triton_impl": { ... },
        "tilelang_impl": {
          "correct": null,
          "status": "not_implemented",
          "reason": "Awaiting implementation"
        }
      }
    }
  ]
}
```

### 8.3 `results/summary.csv`

Flat CSV for pandas/spreadsheet analysis:

```
kernel,config_name,dtype,impl,correct,median_us,peak_allocated_mb,achieved_tflops,sol_compute_pct,achieved_bw_gb_s,sol_memory_pct,bottleneck,speedup_vs_eager
matmul,prefill,fp16,pytorch_eager,true,245.3,128.0,278.4,89.2,,,compute,1.00
matmul,prefill,fp16,pytorch_compile,true,198.7,130.0,343.8,110.2,,,compute,1.23
matmul,prefill,fp16,triton_impl,true,210.4,129.0,324.7,104.1,,,compute,1.17
matmul,prefill,fp16,tilelang_impl,,,,,,,,,
```

Empty fields for unimplemented entries. `speedup_vs_eager` =
`pytorch_eager.median_us / impl.median_us`.

---

## 9. `analyze.py` — Post-hoc Analysis

### 9.1 Interface

```bash
# Generate summary table across all kernels
python -m viperbench.analyze --summary

# Generate per-category breakdown
python -m viperbench.analyze --category matmul

# Generate roofline plot for a specific kernel
python -m viperbench.analyze --roofline matmul --hardware configs/hardware/a100_80gb.json

# Compare two runs (e.g., different hardware)
python -m viperbench.analyze --compare run_a100/ run_h100/

# Export paper-ready tables
python -m viperbench.analyze --latex-tables --output tables/
```

### 9.2 Generated Outputs

| Output | Description |
|---|---|
| `summary_table.csv` | Per-kernel speedup of each impl vs eager, aggregated across configs |
| `per_config_table.csv` | Full detail: every (kernel, config, impl) row |
| `roofline_*.png` | Roofline plots per category: achieved FLOPS/BW vs arithmetic intensity |
| `memory_comparison.csv` | Peak memory per impl, showing DSL memory overhead |
| `coverage_matrix.csv` | Which kernels have implementations in which DSLs |
| `bottleneck_analysis.csv` | Per-kernel classification: compute-bound vs memory-bound per config |
| `shape_sensitivity.csv` | How speedup varies across shape configs for each kernel |

### 9.3 Key Analysis Questions

The analysis module is designed to answer:

1. **Per-DSL headline:** What is the geometric mean speedup of Triton /
   TileLang / torch.compile vs PyTorch eager across all kernels?

2. **Per-category breakdown:** Which kernel categories does each DSL
   excel at? Where does it fall behind?

3. **Shape sensitivity:** For which shape regimes (decode vs prefill vs
   irregular) does the DSL gap change most?

4. **Memory overhead:** Do DSL implementations use more memory than
   PyTorch eager? How much workspace do they allocate?

5. **SOL analysis:** How close does each implementation get to the
   hardware ceiling? Is the gap compute-bound or memory-bound?

6. **Coverage gaps:** Which kernels are NOT_IMPLEMENTED in each DSL?
   What categories are systematically missing?

---

## 10. `utils.py` — Shared Helpers

### 10.1 Dtype Mapping

```python
DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}

BYTES_PER_ELEMENT = {
    "fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
}
```

### 10.2 Hardware Detection

```python
def detect_hardware() -> dict:
    """
    Auto-detect current GPU and load matching hardware config.
    Falls back to generic config if no match found.
    """
    gpu_name = torch.cuda.get_device_name(0)
    # Match against configs/hardware/*.json
    ...

def load_hardware_config(path: Path) -> dict:
    """Load a hardware spec JSON."""
    ...
```

### 10.3 Module Import

```python
def import_module_from_path(path: Path):
    """Import a Python module from a filesystem path."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

### 10.4 Logging

```python
def setup_logger(kernel_name: str, level: str = "INFO") -> logging.Logger:
    """
    Per-kernel logger that writes to both console and
    results/<kernel>/eval.log
    """
    ...
```

---

## 11. Error Handling

### 11.1 Implementation Errors

| Error type | Handling |
|---|---|
| `NOT_IMPLEMENTED = True` | Log as "unsupported", skip, record in results |
| Import error (syntax, missing dep) | Log as "import_error", skip |
| Runtime exception during correctness | Log as "runtime_error", record traceback |
| Correctness failure | Log details, skip profiling |
| OOM during profiling | Log as "oom", record available memory |
| Timeout (kernel hangs) | Kill after configurable timeout (default 60s) |

### 11.2 Partial Results

The runner never fails the entire suite because one kernel errors. Each
kernel's results are written independently. The analysis module handles
missing data gracefully (empty cells in tables, omitted points in plots).

### 11.3 Resumption

The runner checks `results/timing.json` before starting. If a (kernel,
config, impl) result already exists, it skips re-evaluation. Use
`--force` to re-run everything.

---

## 12. Configuration Reference

### 12.1 `configs/defaults.json`

```json
{
  "profiling": {
    "warmup_iters": 10,
    "timed_iters": 100,
    "clear_l2_cache": true,
    "timeout_seconds": 60,
    "compile_warmup_iters": 3
  },
  "correctness": {
    "default_atol": {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-5},
    "default_rtol": {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-5},
    "reference_dtype": "fp32"
  },
  "runner": {
    "default_sweep_mode": "prioritized",
    "max_configs_per_kernel": 1000,
    "log_level": "INFO"
  }
}
```

Per-kernel `input_config.json` overrides these defaults.

### 12.2 Hardware Config Fields

```json
{
  "name": "NVIDIA A100-SXM4-80GB",
  "compute_capability": "sm_80",
  "peak_fp16_tflops": 312,
  "peak_bf16_tflops": 312,
  "peak_fp32_tflops": 156,
  "peak_fp64_tflops": 19.5,
  "peak_int8_tops": 624,
  "memory_bandwidth_gb_s": 2039,
  "memory_capacity_gb": 80,
  "l2_cache_mb": 40,
  "sm_count": 108,
  "max_shared_memory_per_sm_kb": 164
}
```

---

## 13. Adding a New Kernel Checklist

Standard kernel (uses central input_gen + metrics):

```
1. mkdir kernels/<name>
2. Write metadata.json
3. Write input_config.json (use category schema from SPEC)
4. Write reference.py with reference(inputs) -> outputs
5. Create triton_impl.py stub (NOT_IMPLEMENTED = True)
6. Create tilelang_impl.py stub (NOT_IMPLEMENTED = True)
7. Test: python -m viperbench.runner --kernel <name> --correctness-only
```

Nonstandard kernel (needs custom input/metrics):

```
1-6. Same as above
7. Write kernels/<name>/input_gen.py with generate(config, dtype) -> tensors
8. Write kernels/<name>/metrics.py with compute_flops(config) and
   compute_bytes(config, dtype)
9. Test: python -m viperbench.runner --kernel <name> --correctness-only
```