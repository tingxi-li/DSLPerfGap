# Cliff set — provenance of every naive / optimized kernel file

This directory holds the **dual-variant** kernel sources consumed by
`experiments/exp_cliff_roofline.py`. For each kernel with a genuine
*naive → optimized* story we keep BOTH variants so the harness can time
**NAIVE vs OPTIMIZED vs LIBRARY(PyTorch)** at the large benchmark shape and report
the cliff (`t_naive/t_opt`), the two library-efficiency ratios, and a
baseline-independent roofline fraction.

> **A mislabeled naive/opt pair makes every cliff number wrong.** This file is the
> audit trail. Each row gives the exact source (commit + path) the byte-exact copy
> was taken from. Verified 2026-06-26.

Layout:
```
experiments/cliff/naive/<kernel>_<dsl>.py   # functionally-correct-but-slow variant
experiments/cliff/opt/<kernel>_<dsl>.py     # post-mitigation variant (same DSL)
```
Files are **byte-exact copies** (no edits) so the kernels behave identically to
how they ran in ViperBench / AKO4ALL. Provenance lives here, not in file headers.

## Two provenance regimes

* **Round 1** — the original 5 AKO4ALL campaigns. The optimized kernel was written
  *out of tree* (it lives in `AKO4ALL/results/optimized/`), so the ViperBench impl
  was **never modified in place** and is *still the naive version* at HEAD.
    * naive = current `ViperBench/<k>/<dsl>_impl.py` (introduced in `7bde14b`, the
      initial artifact commit; `git log` confirms no later edits).
    * opt   = `AKO4ALL/results/optimized/<k>_<dsl>.py` (KernelBench-wrapper format;
      the unified-API fn is inlined and defined at module top level).

* **Round 2** — the 7 "remaining >2×-slow" kernels, optimized **in place** by commit
  `fdf6b6e` ("Optimize 7 remaining >2x-slow ViperBench kernels"). So:
    * naive = `ViperBench/<k>/<dsl>_impl.py` **@ `fdf6b6e~1` = `a72e84b`** (the
      pre-optimization state), extracted with `git show a72e84b:<path>`.
    * opt   = current `ViperBench/<k>/<dsl>_impl.py` at HEAD (the fdf6b6e version;
      `softmax` and `conv2d` tilelang got a *cosmetic* cleanup in `8b4f25c`
      — "Remove dead torch.softmax fallback; clarify conv2d docstring" — which does
      not change the optimized behavior).

## Provenance table (12 kernel × dsl pairs)

| # | kernel | dsl | round | NAIVE source | OPTIMIZED source | bound |
|---|--------|-----|-------|--------------|------------------|-------|
| 1 | layer_norm | tilelang | 1 | `ViperBench/layer_norm/tilelang_impl.py` @HEAD (7bde14b; never opt'd in place) | `AKO4ALL/results/optimized/layer_norm_tilelang.py` | memory |
| 2 | rms_norm | tilelang | 1 | `ViperBench/rms_norm/tilelang_impl.py` @HEAD (7bde14b) | `AKO4ALL/results/optimized/rms_norm_tilelang.py` | memory |
| 3 | argmax | tilelang | 1 | `ViperBench/argmax/tilelang_impl.py` @HEAD (7bde14b) | `AKO4ALL/results/optimized/argmax_tilelang.py` | memory |
| 4 | matmul | triton | 1 | `ViperBench/matmul/triton_impl.py` @HEAD (7bde14b) | `AKO4ALL/results/optimized/matmul_triton.py` | compute |
| 5 | conv2d | triton | 1 | `ViperBench/conv2d/triton_impl.py` @HEAD (7bde14b) | `AKO4ALL/results/optimized/conv2d_triton.py` | compute |
| 6 | max_reduction | tilelang | 2 | `ViperBench/max_reduction/tilelang_impl.py` @`a72e84b` (=fdf6b6e~1) | `ViperBench/max_reduction/tilelang_impl.py` @HEAD (fdf6b6e) | memory |
| 7 | mean_reduction | tilelang | 2 | `ViperBench/mean_reduction/tilelang_impl.py` @`a72e84b` | `…/mean_reduction/tilelang_impl.py` @HEAD (fdf6b6e) | memory |
| 8 | softmax | tilelang | 2 | `ViperBench/softmax/tilelang_impl.py` @`a72e84b` | `…/softmax/tilelang_impl.py` @HEAD (fdf6b6e + 8b4f25c cleanup) | memory |
| 9 | log_softmax | tilelang | 2 | `ViperBench/log_softmax/tilelang_impl.py` @`a72e84b` | `…/log_softmax/tilelang_impl.py` @HEAD (fdf6b6e) | memory |
| 10 | max_reduction | triton | 2 | `ViperBench/max_reduction/triton_impl.py` @`a72e84b` | `…/max_reduction/triton_impl.py` @HEAD (fdf6b6e) | memory |
| 11 | batched_matmul | tilelang | 2 | `ViperBench/batched_matmul/tilelang_impl.py` @`a72e84b` | `…/batched_matmul/tilelang_impl.py` @HEAD (fdf6b6e) | memory |
| 12 | conv2d | tilelang | 2 | `ViperBench/conv2d/tilelang_impl.py` @`a72e84b` | `…/conv2d/tilelang_impl.py` @HEAD (fdf6b6e + 8b4f25c cleanup) | compute |

The library reference for every row is `ViperBench/<k>/pytorch_impl.py` (loaded by
the harness via `_harness.load_impl(k, "pytorch")`), called with the same inputs.

## What distinguishes naive from optimized (spot-check evidence)

* **layer_norm / rms_norm (tilelang):** naive = `T.serial` mean/var (sum-of-squares)
  loops + fp32 host upcast (`.float()`) + `torch.zeros` output; opt = `T.reduce` +
  native bf16/fp16 I/O + `torch.empty`.
* **argmax / max_reduction / mean_reduction (tilelang):** naive = 3-D `T.serial`
  element scan with fp32 upcast; opt = tiled `T.reduce` reading native dtype
  (max_reduction opt is a two-pass value-then-first-index kernel — `_max_val_kernel`
  / `_first_idx_kernel`, absent from the naive file).
* **softmax / log_softmax (tilelang):** opt = shared-memory row cache + tiled
  reductions + native dtype; naive = whole-row fp32 fragment kernel (log_softmax) or
  a `torch.softmax` fallback (softmax — see caveat below).
* **max_reduction (triton):** naive loads the whole `next_pow2(N)` row in one block
  (spills at N=32768); opt streams `BLOCK_N=4096` tiles with a running argmax.
* **batched_matmul (tilelang):** naive = scalar `T.serial` multiply-accumulate +
  fp32 upcast; opt = shared-A cache + tiled `T.reduce` + native fp16.
* **matmul (triton):** naive = fixed 64×64×64 blocks, no autotune; opt = `@autotune`
  over 12 configs + `GROUP_SIZE_M` L2 swizzle.
* **conv2d (triton):** naive = direct h/w/c loops; opt = padded implicit GEMM +
  autotune + fp16 tensor cores. **conv2d (tilelang):** naive = im2col (`F.unfold`) +
  per-batch TileLang GEMM loop; opt = direct implicit conv as `KH*KW` accumulating
  fp16 tensor-core GEMMs over a spatially-padded input.

## Shapes (single source of truth) & two documented deviations

Shapes/dtypes/fn-names come from **`AKO4ALL/prepare_kernel.py` `KERNEL_CONFIGS`**.

1. **conv2d padding (both dsls): timed with `padding=1`.** `KERNEL_CONFIGS`'
   `get_inputs` passes only `[x, w]` (→ `padding=0`, `OW=126`). That does **not**
   trigger the optimized TileLang conv fast path (gated on `OW % 16 == 0 and
   OW <= 128`) and would yield a spurious cliff ≈ 1.0. The canonical conv-large in
   `ViperBench/benchmark.py:129` uses **`padding=1`** (→ `OW=128`), which is what the
   AKO/round-2 campaigns measured. The harness therefore calls
   `conv2d(x, w, bias=None, stride=1, padding=1)`.

2. **softmax naive = PyTorch at the large shape (read the `note` column).** The naive
   softmax falls back to `torch.softmax(x.float(), …)` for `N > 8192`. At the large
   shape (`N=32768`) the "naive" path is thus a **PyTorch call with an fp32 upcast,
   not a TileLang kernel**. The cliff is still a valid *naive-impl vs optimized-impl*
   ratio, but for softmax it is **not** a "slow `T.serial` DSL kernel" cliff — it is
   a "naive author punted to torch with an fp32 upcast" cliff. Do not over-claim.

## Roofline methodology (the baseline-independent leg)

`roofline_frac_opt = essential_work / (t_opt × peak)`, computed on the **optimized**
kernel only. `essential_work` is the **minimal** DRAM traffic (read inputs once +
write outputs once) for memory-bound ops, or the exact math FLOPs for compute-bound
ops — i.e. the cost of an *ideal* algorithm, so the fraction is a conservative,
library-independent quality bar. Per-kernel formulas live in
`essential_work()` and are echoed into each row's `note`.

Bound classification: norms/reductions/softmax-family/argmax and `batched_matmul`
(GEMV-like, arithmetic intensity ≈ 1 FLOP/B) are **memory-bound**; `matmul` and
`conv2d` are **compute-bound** (fp16 tensor cores).

Per-GPU peaks (documented dense vendor specs; no sparsity; FP16/BF16 with FP32
accumulate), matched by substring of the runtime device name:

| GPU (slug substring) | HBM BW (memory peak) | FP16/BF16 TC (compute peak) |
|---|---|---|
| **GH200** (run target; `NVIDIA_GH200_480GB`) | **4.0 TB/s** (96 GB HBM3; 144 GB HBM3e SKU ≈ 4.9 TB/s) | **989.5 TFLOP/s** (Hopper dense) |
| H100 (`NVIDIA_H100_80GB_HBM3`) | 3.35 TB/s | 989.5 TFLOP/s |
| A100 (`NVIDIA_A100-*-40GB`) | 1.555 TB/s (40 GB; 80 GB SXM ≈ 2.039) | 312 TFLOP/s |

If the runtime GPU matches no entry, the harness writes the row with
`roofline_frac_opt=""` and a `note` explaining the gap (it never guesses a peak).
Add an entry to `PEAKS` in `exp_cliff_roofline.py` before trusting a new GPU's
roofline fraction.

## Not used (and why)

* `experiments/opt_kernels/{softmax,log_softmax,max_reduction,mean_reduction,
  logsumexp}_opt.py` — earlier **standalone** benchmark scripts (commit `637021f`)
  with CUDA-libpath re-exec shims and inline `import time` timing. They are *not*
  importable unified-API impls and were **superseded** by the in-place fdf6b6e
  ViperBench versions, which are the canonical round-2 optimized kernels used here.
* `logsumexp` — has an `opt_kernels` draft but **no committed naive→opt pair** in
  ViperBench (not touched by `fdf6b6e`), so it is excluded from the cliff set.
