# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Research artifact for ASE 2026 paper #4134 (`ase26-paper4134.pdf`, reviewer responses in `reviews.txt`). It has two coupled subsystems:

- **ViperBench/** — a benchmark suite comparing **PyTorch**, **Triton**, and **TileLang** implementations of 22 deep-learning kernels for both correctness and latency/memory.
- **AKO4ALL/** — an agentic kernel-optimization loop: a coding agent (Claude Code) iteratively rewrites a single kernel for maximum speed under a strict protocol.

The two are bridged by `AKO4ALL/prepare_kernel.py`, which exports a ViperBench kernel into AKO4ALL's optimization format.

> The most useful starting points are `README.md` (ViperBench + AKO4ALL overview) and `AKO4ALL/README.md` (the optimizer in depth).

## ⚠️ Stale root-level scaffolding — ignore it

The repo root contains an **abandoned earlier scaffold** that does *not* run against the real project: `run_all.py`, `profile_all.py`, `test_harness.py`, `test_kernel.py`, `tilelang_impl.py`, `pytorch_ref.py`, `bench_gemm_quick.py`, and `tests/results/`. These hardcode `KERNELS_DIR = Path("newBench")` — **a directory that does not exist**. Do not run, edit, or copy patterns from them. The live equivalents all live under `ViperBench/` (e.g. the real runner is `ViperBench/run_all.py`, the real harness is `ViperBench/test_utils.py`). Treat the root scaffold as dead code unless a task explicitly targets it.

## Common commands

```bash
pip install torch triton tilelang        # ncu (Nsight Compute) also needed for AKO4ALL profiling

# Correctness
python ViperBench/run_all.py              # run every ViperBench/<kernel>/test.py, print summary
python ViperBench/<kernel>/test.py        # one kernel (e.g. layer_norm); exits 0 pass / 1 fail

# Latency + peak-memory benchmarks → ViperBench/results/profile.csv
python ViperBench/benchmark.py            # PyTorch + Triton (+ *_tuned variants)
python ViperBench/benchmark_tilelang.py   # TileLang
python ViperBench/benchmark_tuned.py      # all impls including tuned

# Auto-tuning sweep (writes results/tuning_cache.json) — run as a module from inside ViperBench/
cd ViperBench && python -m tuning.sweep --all
cd ViperBench && python -m tuning.sweep --kernel matmul --impl triton

# AKO4ALL: optimize one kernel
cd AKO4ALL && python prepare_kernel.py <kernel> <triton|tilelang>   # export from ViperBench
bash scripts/bench.sh baseline            # verify CORRECT=True before optimizing
cd AKO4ALL && claude                      # then: "Follow the instructions in TASK.md."
```

There is no build step or linter; "tests" means the per-kernel correctness scripts above.

## ViperBench architecture

Each kernel lives in `ViperBench/<kernel>/` with exactly four files:

| File | Role |
|------|------|
| `pytorch_impl.py` | reference (torch / cuDNN built-ins) — the correctness golden |
| `triton_impl.py`  | custom Triton kernel |
| `tilelang_impl.py`| custom TileLang kernel |
| `test.py`         | wires all three into the shared harness and `sys.exit()`s |

**Unified-API contract (the key invariant).** All three `*_impl.py` files export a function with the **same name as the kernel directory** and the **same signature**, so they are drop-in interchangeable (e.g. every backend defines `layer_norm(x, weight, bias, eps=1e-5)`). The PyTorch reference deliberately *raises* on argument values the hand-written kernels don't support (e.g. `layer_norm` rejects `eps != 1e-5` because the Triton kernel hardcodes it) — this keeps the three backends locked to identical behavior. **If you add or change a kernel, all three signatures must stay aligned or tests break.**

**Test harness** (`ViperBench/test_utils.py`):
- `run_test(...)` compares PyTorch vs Triton; `run_tilelang_test(...)` compares PyTorch vs TileLang. `test.py` picks one. Both write JSON to `ViperBench/results/<kernel>[_tilelang].json` and call `sys.exit(0|1)` — so `test.py` runs on import; always invoke it as a subprocess, never import it.
- Tolerances by dtype: fp32 `1e-5`, fp16 `1e-3`, bf16 `1e-2`. Pass `loose_tol=True` (used for reductions / normalizations) to double them for order-of-operations drift.
- A `test.py` defines `test_cases` (list of `{"name", "inputs", "dtype"}`); `inputs` may be a tuple/list (positional) or dict (keyword).

**Tuning is loaded at import time — and silently arch-dependent.** Every `*_impl.py` begins with:

```python
from tuning.cache import get_best_config as _get_best_config
_TUNED = _get_best_config("<kernel>", "<impl>") or {}   # falls back to {} on any error
```

then reads block sizes / thread counts / num_stages from `_TUNED` with hardcoded defaults (`_TUNED.get("BLOCK_SIZE_M", 64)`). `cache.py` keys configs as `"<kernel>/<impl>/<gpu_arch>"` in `ViperBench/results/tuning_cache.json`, where `gpu_arch` comes from `torch.cuda.get_device_name(0)`. **Consequence:** on a GPU/arch with no cached entry you transparently get the defaults — kernel correctness never changes, but performance does. The candidate grids per kernel live in `ViperBench/tuning/configs.py` (`TRITON_CONFIGS`, `TILELANG_CONFIGS`); `sweep.py` times them and writes the winners back to the cache.

**Benchmarking.** `benchmark*.py` import each impl via `importlib`, run CUDA-synchronized `perf_counter` timing (10 warmup / 100 measured, median) plus `torch.cuda.max_memory_allocated()`, and append rows to `results/profile.csv` with columns `kernel,size,impl,input_desc,latency_ms,peak_memory_mb`. `impl` ∈ {`pytorch`, `triton`, `triton_tuned`, `tilelang`, `tilelang_tuned`}. Input shapes come from `kernel_input_shapes.html`. `results/slow_kernels.csv` lists kernels slower than PyTorch on large inputs.

## AKO4ALL architecture

`prepare_kernel.py <kernel> <impl>` is the ViperBench→AKO4ALL bridge: it reads `ViperBench/<kernel>/pytorch_impl.py` (golden) and `<impl>_impl.py` (target), and emits `input/reference.py` + `solution/kernel.py` in **KernelBench format** plus `scripts/bench.sh`. Per-kernel large-input shapes are hardcoded in its `KERNEL_CONFIGS` dict (only kernels listed there can be exported this way).

The optimization loop is governed by two files a running agent must obey **exactly**:
- `TASK.md` — rigid protocol: analyze `input/`+`context/`+`bench/`+`HINTS.md` → create an `opt/<kernel>` git branch → copy kernel to `solution/` → generate `scripts/bench.sh` (fill `{{BENCH_COMMAND}}` in `bench-wrapper.sh`) → verify baseline `CORRECT=True` and commit. Then iterate: **one iteration = one code edit + one `bash scripts/bench.sh iter-N` run**; after each, update `ITERATIONS.md` and `git commit -m "[iter N] ..."`. Goal is *genuine* latency reduction — reward hacking (stream injection, timing monkey-patching, uninitialized output) is forbidden.
- `HINTS.md` — constraints: run `ncu` before iteration 1 and re-profile after 3 non-improving iterations; optimize for large inputs; **never switch language** (TileLang stays TileLang, Triton stays Triton — no PyTorch/cuDNN fallback); do not install packages.

Other pieces:
- `bench/kernelbench/` — built-in correctness+timing evaluator (used when `bench/` has nothing else). Anti-cheat: warns on >10× speedup and overrides the solution's `get_inputs`/`get_init_inputs` with the reference's.
- `context/` — reference docs the agent may consult: `GPU Kernel Performance Analysis Report.md`, `tilelang_reference.md`, `triton_tuning.md`, `known_github_issues.md`.
- `results/optimization_results.csv` + `results/optimized/<kernel>_<impl>.py` + `<kernel>_iterations.md` — completed campaigns (e.g. layer_norm TileLang 1090ms→0.89ms = 1224×). The dominant TileLang win pattern there: replace `T.serial` reduction loops with `T.reduce`, do native-dtype I/O, and use `torch.empty` over `torch.zeros`.
- `.gitignore` excludes per-run AKO4ALL artifacts: `scripts/`, `trajectory/`, `_bench_output.txt`, `.claude/`. Note the artifact root itself is **not** a git repo — the AKO4ALL workflow initializes/uses git only within its own run.

## Working in this repo

- Adding/modifying a kernel: keep the three backends' function name and signature identical; add `test_cases` covering small/medium/large/edge shapes; use `loose_tol=True` for reductions. Register tuning grids in `tuning/configs.py` if the kernel takes config params.
- TileLang gotcha worth remembering (it drove the biggest speedups here): `T.serial` reduction loops are slow — prefer `T.reduce`; use `T.Pipelined(num_stages=...)` to hide memory latency, `T.gemm` for matmul-heavy ops, and `T.use_swizzle` for L2 locality. Always `T.clear()` an accumulator before `T.gemm`. Deeper notes: `AKO4ALL/context/tilelang_reference.md` and `triton_tuning.md`.
