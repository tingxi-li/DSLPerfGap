# TileLang Kernel Porting Project

## Project Overview
This project ports GPU kernels from PyTorch/Triton implementations to TileLang.
Each kernel must produce **numerically identical outputs** across all three implementations.

## Directory Structure
```
newBench/
  <kernel_name>/
    *.py                # PyTorch reference implementation (non-triton file)
    triton*.py          # Triton implementation (filename starts with "triton")
    tilelang_impl.py    # TileLang implementation — YOU CREATE THIS
    test_kernel.py      # Unified test harness — YOU CREATE THIS
tests/
  results/              # JSON test result logs
run_all.py              # Runs all kernels sequentially (project root)
test_harness.py         # Shared test utilities (project root)
```

## Kernel Discovery
Run this to find all kernels before starting:
```bash
ls newBench/
```
For each subdirectory in `newBench/`:
- The **PyTorch reference** = the `.py` file whose name does NOT start with `triton`
- The **Triton implementation** = the `.py` file whose name starts with `triton`
- You must create `newBench/<name>/tilelang_impl.py`
- You must create `newBench/<name>/test_kernel.py`

When importing in `test_kernel.py`, use relative imports or `sys.path.insert(0, os.path.dirname(__file__))` to load siblings from the same `newBench/<name>/` directory. Also add the project root to sys.path to import `test_harness`.

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
1. `ls newBench/<n>/` to identify the pytorch ref file and triton file
2. Read both files fully to understand the algorithm, shapes, and dtypes used
3. Create `newBench/<n>/tilelang_impl.py`
4. Create `newBench/<n>/test_kernel.py` (import test_harness from project root)
5. Run: `python newBench/<n>/test_kernel.py`
6. If test fails, read the full error, fix `tilelang_impl.py`, retry (up to **20 attempts**)
7. On pass: log result to `tests/results/<kernel_name>.json`, move to next kernel

### Test Harness Template (`test_kernel.py`)
Each test file must:
- Test at least **4 shapes** (small, medium, large, edge-case)
- Use `torch.allclose(atol=1e-3, rtol=1e-3)` for fp16; `atol=1e-5` for fp32
- Print PASS/FAIL per shape with max absolute error
- Exit with code 0 on all-pass, 1 on any failure

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
- After all kernels: run `python run_all.py` for a final summary report

## Numerical Correctness Criteria
| dtype   | atol   | rtol   |
|---------|--------|--------|
| float32 | 1e-5   | 1e-5   |
| float16 | 1e-3   | 1e-3   |
| bfloat16| 1e-2   | 1e-2   |

Reductions (softmax, layernorm) may use slightly looser tolerances (2× above) due to order-of-operations differences.

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