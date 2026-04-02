# TileLang Kernel Porting Project

## Project Overview
This project ports GPU kernels from PyTorch/Triton implementations to TileLang.
Each kernel must produce **numerically identical outputs** across all three implementations.

## Directory Structure
```
KernelBench_dedup/
  level1/                   # 100 primitive ops (matmul, relu, conv, etc.)
    <kernel_name>/
      pytorch_impl.py       # PyTorch reference + unified harness
  level2/                   # 99 fused operation chains (conv+relu+bn, etc.)
    <kernel_name>/
      pytorch_impl.py
  level3/                   # 50 complete models (ResNet, VGG, LSTM, etc.)
    <kernel_name>/
      pytorch_impl.py
  tritonbench/              # 48 optimized Triton operator benchmarks
    <kernel_name>/
      operator.py           # Original tritonbench operator (read-only reference)
      pytorch_impl.py       # Standalone PyTorch adapter with unified harness
  categories/               # Kernel categorization (22 categories)
    categories.json         # Master category mapping
    by_category/            # Per-category kernel lists
    CHANGES.md              # Corrections from original categorization
  eval_all.py               # Unified eval harness for all 297 kernels
  scripts/
    add_unified_interface.py  # Automation script (appends harness to level1/2/3)
  results/                  # JSON eval output
```

## Unified Kernel Interface

Every `pytorch_impl.py` exposes the same interface:

```python
class Model(nn.Module):
    def __init__(self, ...):    # Constructor (args from get_init_inputs())
        ...
    def forward(self, ...):     # The kernel computation
        ...

def get_inputs():               # Hardcoded test inputs (CPU tensors)
    return [tensor1, tensor2, ...]

def get_init_inputs():           # Model constructor arguments
    return [arg1, arg2, ...]

def get_test_inputs():           # CUDA-ready inputs
    return [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]

def run(*args):                  # One-call entry point
    if args:
        inputs = args
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
```

**To run any kernel:** `import pytorch_impl; output = pytorch_impl.run()`

## Kernel Discovery
```bash
# List all 297 kernels
python KernelBench_dedup/eval_all.py --list

# Run all kernels
python KernelBench_dedup/eval_all.py

# Filter by level or kernel name
python KernelBench_dedup/eval_all.py --level level1
python KernelBench_dedup/eval_all.py --kernel softmax
python KernelBench_dedup/eval_all.py --level tritonbench --kernel gemm
```

## Kernel Categories (22 total, 297 kernels)

**Primitive Operations (level1 + tritonbench):**
matmul (21), conv (35), activation (17), normalization (11), pooling (6),
reduction (7), loss (11), attention (10), cumulative (5), embedding (2),
dropout (1), quantization (10), elementwise (3), specialized (9)

**Fused Operations (level2):**
fused_conv (33), fused_convtranspose (30), fused_gemm (26), fused_matmul (10)

**Complete Models (level3):**
model_cnn (25), model_transformer (8), model_rnn (10), model_other (7)

See `KernelBench_dedup/categories/categories.json` for the full mapping.

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
1. Pick a kernel from `KernelBench_dedup/` (any level)
2. Read `pytorch_impl.py` to understand the algorithm, shapes, and dtypes
3. Create `tilelang_impl.py` in the same directory
4. Create `test_kernel.py` that compares PyTorch vs TileLang outputs
5. Run: `python KernelBench_dedup/<level>/<kernel>/test_kernel.py`
6. If test fails, read the full error, fix `tilelang_impl.py`, retry (up to **20 attempts**)
7. On pass: log result, move to next kernel

### Retry Strategy
When a TileLang kernel fails:
1. Print the full error/traceback
2. Diagnose: shape mismatch? wrong indexing? wrong dtype? missing `T.clear()`?
3. Edit `tilelang_impl.py` with a targeted fix
4. Re-run test immediately
5. Repeat up to 20 times; if still failing after 20, write a `FAILED` entry to results and move on

## Autonomy Rules
- **Always allowed**: read/write files in `./`, run `python`, `grep`, `nvidia-smi`, `pip install`
- **Blocked**: `sudo`, network calls outside pip installs
- Work sequentially through kernels; do not skip ahead unless a kernel exceeds 20 retries
- After all kernels: run `python KernelBench_dedup/eval_all.py` for a final summary report

## Numerical Correctness Criteria
| dtype   | atol   | rtol   |
|---------|--------|--------|
| float32 | 1e-5   | 1e-5   |
| float16 | 1e-3   | 1e-3   |
| bfloat16| 1e-2   | 1e-2   |

Reductions (softmax, layernorm) may use slightly looser tolerances (2x above) due to order-of-operations differences.

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

## Common TileLang Pitfalls
- Always call `T.clear(C_local)` before accumulation
- Block tile sizes must divide evenly into tensor dims, or use padding
- `T.copy` direction: `T.copy(global_src[offset], shared_dst)` — order matters
- `T.gemm` expects (A, B, C) where C is the accumulator (fragment, not shared)
- For elementwise ops, iterate with `for i, j in T.Parallel(M, N):` or use thread indices
- `threads=` in `T.Kernel` must be a multiple of 32 (warp size)
- Output `T.copy` from fragment to global: `T.copy(C_local, C[by*bM, bx*bN])`
