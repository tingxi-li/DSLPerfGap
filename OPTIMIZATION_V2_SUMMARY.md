# Kernel optimization round 2 — remaining >2× slow kernels

Optimizes the ViperBench kernels that were **>2× slower than PyTorch on large
inputs** and were *not* among the 5 original AKO4ALL campaigns
(`AKO4ALL/results/optimized/`). Seven `(kernel, backend)` targets, both Triton
and TileLang. Optimizations are applied **in-place** in
`ViperBench/<kernel>/{triton,tilelang}_impl.py`; all changes keep a correct
fallback for the small/edge test shapes, so every `ViperBench/<kernel>/test.py`
still passes.

Measured on **NVIDIA GH200 (Hopper)** — note this differs from the Ada GPU in
`ViperBench/results/slow_kernels.csv`, so PyTorch baselines are much faster here
(huge HBM3 bandwidth) and the gaps are re-baselined on this GPU.

| # | kernel | backend | before | gap | after | gap now | speedup |
|---|--------|---------|-------:|----:|------:|--------:|--------:|
| 1 | max_reduction  | tilelang | 5.26 ms | 18.7× | 0.33 ms | 1.18× | **15.8×** |
| 2 | mean_reduction | tilelang | 4.31 ms | 14.2× | 0.32 ms | parity | **13.6×** |
| 3 | softmax        | tilelang | 1.16 ms | 4.6× | 0.30 ms | 1.15× | **3.9×** |
| 4 | log_softmax    | tilelang | 1.21 ms | 5.7× | 0.24 ms | match | **5.0×** |
| 5 | max_reduction  | triton   | 1.50 ms | 5.3× | 0.24 ms | **0.85× (faster)** | **6.3×** |
| 6 | batched_matmul | tilelang | 6.28 ms | 17.8× | 0.32 ms | **0.90× (faster)** | **19.8×** |
| 7 | conv2d         | tilelang | 18.58 ms | 14.6× | 2.29 ms | 1.61× | **8.1×** |

Every target now runs **under 2× of PyTorch** (5 at parity or faster). Full
before/after with input shapes and strategies: `AKO4ALL/results/optimization_results_v2.csv`.

## Techniques

**TileLang reduction family** (max/mean/softmax/log_softmax) — the same lesson
as the original norm campaigns: replace `T.serial` reduction loops with parallel
`T.reduce`, and read/write in the native dtype instead of upcasting to fp32.
- Reductions (max, mean): one block per row, stream the row in power-of-2 tiles,
  `T.reduce` each tile, accumulate. `max_reduction` returns value **and** index,
  so it runs two bandwidth-bound passes — `reduce max` for the value, then
  `reduce min` over candidate indices for the first-occurrence argmax (matches
  `torch.max` tie-breaking exactly).
- softmax/log_softmax: one block per row with the row cached in **shared memory**
  (1 global read + 1 global write), tiled `T.reduce` for the max and sum. This
  removes the old PyTorch fallback for `N > 8192` (which both violated the
  no-fallback rule and was slow due to an fp16→fp32 round-trip).

**batched_matmul (TileLang)** — `C[m,n]=Σ_k A[m,k]B[m,n,k]` is bandwidth-bound on
reading B (1 GB fp16). Cache `A[m,:]` in shared once, stream B in `(bN,bK)` tiles
via `T.copy`, multiply by the A slice and `T.reduce` over K. The old kernel
upcast B to fp32 (2× the bytes) and read it element-by-element in a serial loop.

**max_reduction (Triton)** — the old kernel loaded the entire `next_pow2(N)` row
in a single block (catastrophic spill for N=32768). Replaced with a tiled
streaming reduction keeping a running max + argmax across `BLOCK_N`-wide tiles.

**conv2d (TileLang)** — the hard one (cuDNN is heavily optimized on Hopper). The
old im2col path materialized a ~4.8 GB fp32 column tensor and looped per batch.
Replaced with a **direct implicit-conv**: pad the input spatially once, then
express the conv as `KH*KW` accumulating fp16 **tensor-core GEMMs** (one per
kernel tap). With a padded input each tap's input tile is a contiguous,
bounds-free, coalesced affine slice — no im2col, no per-element index math. The
kernel writes straight into NCHW layout to avoid a post-kernel transpose. A
hoisted-index implicit GEMM and a single-big-GEMM im2col were both tried and were
slower (see notes below). Gated to stride-1, tensor-core-friendly configs; other
configs use the original im2col path.

### conv2d: what didn't work
- **Single big fp16 GEMM via `F.unfold`** — 9.2 ms; the unfold + permute copies
  dominate.
- **Implicit GEMM with per-element `(c,kh,kw,n,oh,ow)` decode** — 9.5 ms naive,
  4.5 ms with the index decode hoisted; the masked gather cannot use TileLang's
  `T.Pipelined` async-copy stages (`ProducerConsumerWS` / role-assignment errors),
  so the K-loop stays serial.
- The winning direct-GEMM form sidesteps all of that by padding the input so the
  per-tap load is a plain affine copy.

## Environment note
TileLang was not installed in this environment. On this aarch64 / Grace-Hopper
box the working install is: `pip install tilelang==0.1.11` then pin
`numpy<2` (1.26.x) and `apache-tvm-ffi==0.1.11` — tvm-ffi 0.1.12 aborts with a
double FFI-type-registration crash, and numpy 2.x breaks `ml_dtypes`.
