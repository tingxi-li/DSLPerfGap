# ASE 2026 Paper #4134 — Artifact

> *"An Empirical Study of GPU Kernel Performance Gaps in Modern Domain-Specific Languages."*

 Each reviewer concern addressed this round maps to one or more files under `experiments/results/<gpu>/` (with `<gpu> = NVIDIA_RTX_4000_Ada_Generation`) or `ViperBench/results/`.

Throughout, **E_lib %** = library-efficiency (higher is better; PyTorch / cuBLAS / cuDNN = 100%).

---

## Rebuttal Deliverables — what was completed and what it shows

Items 1–10 below are completed. Item 11 is the one remaining experiment.

### 1. Hardware-counter evidence + RC2b / RC3 / RC4 isolations

**Evidence:** [`NCU_FINDINGS.md`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/NCU_FINDINGS.md) · [`ncu_summary.csv`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/ncu_summary.csv) · [`winograd_isolation.csv`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/winograd_isolation.csv) · [`cudnn_winograd_3x3.log`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/cudnn_winograd_3x3.log)

**Counter table (one launch per kernel; full 24/24 collections in the supplementary CSV)**

| metric (RC) | matmul cuBLAS | matmul Triton | conv2d Triton | layer_norm TileLang | argmax TileLang | max_reduction TileLang |
|---|--:|--:|--:|--:|--:|--:|
| RC3 regs/thread | 218 | 154 | 128 | **254** | 39 | 39 |
| RC3 achieved occ % | 16.7 | 25.0 | 33.3 | **16.5** | 80.7 | 64.4 |
| RC3 spill load (B) | 0 | 0 | 0 | **51.5 GB** | 0 | 0 |
| RC3 spill store (B) | 0 | 0 | 0 | **34.4 GB** | 0 | 0 |
| RC0 stall long_scoreboard | 0.06 | 22.2 | 10.2 | **104.9** | 27.4 | 86.7 |
| RC0 stall barrier (sync) | 0.09 | 8.02 | 1.54 | **0** | **0** | **0** |
| RC1 global-load eff % | 99.7 | 0\* | **36.4** | 50.0 | 12.5 | 12.5 |
| RC2b L2 hit % | 91.0 | 66.2 | 95.2 | 53.1 | **0.64** | 38.6 |
| RC2b DRAM throughput % | 27.2 | 56.7 | 19.5 | **90.2** | 10.2 | 20.8 |
| kernel time (ms) | 123.6 | 363.1 | 36.6 | 271.2 | 30.9 | 34.1 |

\* Triton matmul reads 0% because it stages tiles via `cp.async` (LDGSTS), which bypasses the LDG counter — not an uncoalesced-load signal.

**RC4 Winograd isolation (cuDNN 3×3 stride-1; determinism A/B)**

| config | mode | median ms | note |
|---|---|--:|---|
| 3×3 s1 | Winograd allowed (nondeterministic) | 10.62 | — |
| 3×3 s1 | Winograd off (deterministic) | 10.86 | upper bound on Winograd benefit |
| **Δ** | det − nondet | **+0.24 ms (≈ 2.3%)** | RC4 ≈ 2–3% of the conv gap |

**What the counters localize:**
- RC0(a) `T.serial` reductions are memory-latency-bound: `long_scoreboard` 27–105, `barrier ≈ 0`. *Kernel-authoring fix: `T.serial → T.reduce`.*
- RC0(b) absent LDG.128 vectorization surfaces as low global-load efficiency (conv 36.4% vs cuBLAS 99.7%) — a code-generation issue, distinct from RC0(a).
- RC3 register pressure: TileLang LayerNorm (254 regs, 86 GB spill traffic); conv `n_spills = 0` — its gap is occupancy + coalescing.
- RC4 Winograd accounts for ≈2.3% of the conv gap (cuDNN determinism A/B).

---

### 2. Convolution coverage + mitigation generality


**Evidence:** [`conv_mitigation_large.csv`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/conv_mitigation_large.csv) (table cells below — baseline + mitigation, single locked-clocks run) · [`conv_filters_large.csv`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/conv_filters_large.csv) (earlier baseline-only sweep) · `_small` variants

**Filter sweep on shape `32×256×128×128`, fp16 (E_lib %, higher is better)**

| filter | stride | groups | PyTorch (ref) | Triton | TileLang | **Mitigation (optimized Triton)** |
|---|---|---|--:|--:|--:|--:|
| 1×1 | 1 | 1 | 100 | 24.0 | 9.9 | **103.6** |
| 3×3 | 1 | 1 | 100 | 34.2 | 11.2 | **69.0** |
| 5×5 | 1 | 1 | 100 | 14.0 | OOM | **61.5** |
| 7×7 | 1 | 1 | 100 | 12.1 | OOM | **57.5** |
| 3×3 | 2 | 1 | 100 | 35.6 | 20.5 | **68.8** |
| 3×3 | 1 | 256 (depthwise) | 100 | 2.7 | 0.4 | *excluded — kernel is `groups==1`* |

The baseline conv gap **widens with filter size** (34% → 12% across 3×3 → 7×7); the optimized implicit-GEMM kernel **holds 57–69% across the family**, all numerically correct. Depthwise is the one stated exclusion. The 1×1 control isolates RC1.

---

### 3. Tuning clarification — §5 ↔ §7.3 reconciliation


**Evidence:** [`autotune_matmul.csv`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/autotune_matmul.csv)

**Matmul autotune — both shapes, same GPU**

| shape | impl (search space) | median ms | GFLOPS | speedup vs plain | **E_vs_cuBLAS** |
|---|---|--:|--:|--:|--:|
| 4096² | cuBLAS | 1.94 | 70 752 | 1.21× | 100% |
| 4096² | Triton plain | 2.34 | 58 713 | 1.00× | **83.0%** |
| 4096² | Triton **autotuned** (expanded) | 1.90 | 72 316 | 1.23× | **102.2%** |
| 4096² | TileLang swizzle | 3.16 | 43 455 | 0.74× | 61.4% |
| 16384² | cuBLAS | 116.63 | 75 416 | 3.10× | 100% |
| 16384² | Triton plain | 361.90 | 24 305 | 1.00× | **32.2%** |
| 16384² | Triton **autotuned** (expanded) | 118.42 | 74 279 | 3.06× | **98.5%** |
| 16384² | TileLang swizzle | 202.60 | 43 416 | 1.79× | 57.6% |

- §5's "heuristic tuning" = 12-config block-tile grid (`bm,bn ∈ {32,64,128}, bk ∈ {32,64}`) → Δ ≈ 0pp (block-tile alone doesn't beat the default).
- §7.3's **expanded search** adds `GROUP_SIZE_M` (L2 swizzle), `num_warps`, `num_stages` → recovery of **32% → 98%** at 16384² and **83% → 102%** at 4096².
- Reconciliation is **search-space**, not just shape; the dramatic recovery is at the RQ1 16384² shape.

---

### 4. Baseline fairness — split metrics


**Evidence:** [`fused_baselines.csv`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/fused_baselines.csv)

**Three baselines side-by-side (large shapes; ms, lower is better)**

| kernel | shape | eager | **fused `torch.compile`** | Triton | TileLang | E_vs_fused (Triton) |
|---|---|--:|--:|--:|--:|--:|
| rms_norm | (8192, 8192) | 9.98 | **1.77** | 0.91 | 187.10 | **194.9%** |
| swiglu | (4096, 32768) | 3.44 | **3.05** | 1.29 | 3.43 | **235.9%** |
| softmax | (4096, 32768) | 1.75 | **3.61** | 1.80 | 8.70 | 200.9% |
| log_softmax | (4096, 32768) | 1.75 | **3.55** | 2.19 | 10.39 | 162.3% |
| add | (67M,) | 1.30 | **3.01** | 1.28 | 1.72 | 235.8% |
| relu | (16384, 16384) | 3.55 | **7.04** | 3.42 | 5.17 | 205.9% |
| leaky_relu | (8192, 8192) | 71.95 | **72.67** | 29.52 | 19.34 | 246.2% |

For LayerNorm specifically the baseline is **already** the fused `F.layer_norm`, so Triton's 94.5% is a fair library comparison. Only RMSNorm and element-wise use eager paths; the per-category split in the revision states this explicitly. `torch.compile` does not always improve over eager (e.g. softmax, add, relu).

---

### 5. FP32 GEMM root-cause


**Evidence:** [`fp32_gemm.csv`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/fp32_gemm.csv) · script [`exp_fp32_gemm.py`](experiments/exp_fp32_gemm.py)

**Four-arm controlled isolation**

| arm | max abs err | max rel err | % mismatch | verdict |
|---|--:|--:|--:|---|
| **A.** `T.gemm` fp32 | 0.176 | 3013 | 31.8 % | FAIL @ 1e-5, **same magnitude class as cuBLAS's own TF32 path** |
| **B.** Manual fp32 MAC (no `T.gemm`) | 4.9e-4 | 4.8 | 0.03 % | true-fp32 class (mean rel 7e-7) → fault is **isolated to `T.gemm`** |
| **C.** A vs cuBLAS-TF32 ref | 0.194 | 645.9 | 42.9 % | A matches the TF32 signature: **definitive TF32 truncation in `T.gemm`** |
| **D.** TF32 knob scan | — | — | — | **No user-space TF32 disable** (`PassConfigKey` + `T.gemm` expose none; fp32 lowers to SM80 `F32TF32TF32F32` MMA) |

The signature — max-rel-err at a near-zero output (cancellation), small mean abs err — confirms TF32 mantissa truncation, **not** a layout/indexing bug. The FP32 failure becomes a first-class SE finding in the revision.

---

### 6. Correctness methodology + edge cases


**Evidence:** [`correctness_edge.csv`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/correctness_edge.csv) · script [`exp_correctness_edge.py`](experiments/exp_correctness_edge.py)

**Edge-case coverage (kernel × input → status)**

| kernel | normal | NaN | +Inf | −Inf | large-mag | denormal | all-equal rows |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| matmul (TileLang) | NUMERIC_DIFF\* | PROPAGATES_SAME | PROPAGATES_SAME | PROPAGATES_SAME | DIVERGE_NAN_INF | MATCH | NUMERIC_DIFF\* |
| add (TileLang) | MATCH | PROPAGATES_SAME | PROPAGATES_SAME | PROPAGATES_SAME | MATCH | MATCH | MATCH |
| softmax (TileLang) | MATCH | PROPAGATES_SAME | PROPAGATES_SAME | PROPAGATES_SAME | MATCH | MATCH | MATCH |
| layer_norm (TileLang) | MATCH | PROPAGATES_SAME | PROPAGATES_SAME | PROPAGATES_SAME | MATCH | MATCH | MATCH |

\* matmul NUMERIC_DIFF is the FP32 root-caused in §5 above (TF32 truncation, not a logic error).

**Mitigation-kernel revalidation (post-optimization correctness)**

| kernel | tolerance | result |
|---|---|---|
| layer_norm (TileLang, optimized) | atol/rtol = 0.02 / 0.02 | REVALIDATED · max_abs 0.0625 |
| rms_norm (TileLang, optimized) | atol/rtol = 0.002 / 0.002 | REVALIDATED · max_abs 0.00391 |
| argmax (TileLang, optimized) | exact index match | REVALIDATED · 100% match |
| matmul (Triton, optimized) | atol/rtol = 0.2 / 0.01 | REVALIDATED · max_abs 0.125 |
| conv2d (Triton, optimized) | atol/rtol = 0.002 / 0.002 | REVALIDATED · max_abs 0.0312 |

Tolerances by dtype (with `loose_tol = 2×` for reductions/normalizations): fp32 1e-5 · fp16 1e-3 · bf16 1e-2.

---

### 7. Per-kernel element-wise breakdown

**Evidence:** [`ViperBench/results/profile.csv`](ViperBench/results/profile.csv)

**Element-wise kernels at large shapes (ms; with tuned variants when present)**

| kernel | input | PyTorch | Triton | Triton tuned | TileLang | TileLang tuned |
|---|---|--:|--:|--:|--:|--:|
| add | (64M,) fp16 | 1.33 | 1.28 | 1.28 | 1.72 | 1.71 |
| mul | (64M,) fp16 | 0.89 | 1.28 | 1.28 | 1.29 | 1.29 |
| relu | (16384,16384) fp16 | 3.54 | 3.42 | 3.42 | 5.15 | 5.15 |
| leaky_relu | (8192,8192) fp16 | 72.04 | 29.46 | 30.00 | 19.67 | 19.74 |
| swiglu | (4096,32768) fp16 | 3.44 | 1.30 | 1.29 | 3.43 | 3.43 |

Within-category spread is large (Triton: 36–366% of PyTorch; TileLang: 27–366%) — the per-category aggregate alone hides where DSLs win (Triton on add/swiglu; TileLang on leaky_relu) vs lose (TileLang on relu).

---

### 8. "Iteration" definition + per-kernel logs

**Evidence:** [`AKO4ALL/results/optimization_results.csv`](AKO4ALL/results/optimization_results.csv) · per-kernel logs `AKO4ALL/results/optimized/<kernel>_iterations.md` · protocol [`AKO4ALL/TASK.md`](AKO4ALL/TASK.md)

**Definition.** *One iteration = one kernel-source edit + one benchmark run + a correctness check, logged and git-committed. Failed and regressing attempts count.*

**Completed campaigns**

| kernel | DSL | input | before | after | speedup gained | iters | strategy |
|---|---|---|--:|--:|--:|--:|---|
| layer_norm | TileLang | x:(8192,8192) bf16 | 1090.0 ms | 0.89 ms | **1224×** | 18 | `T.serial → T.reduce` + native bf16 I/O + `torch.empty` |
| rms_norm | TileLang | x:(8192,8192) fp16 | 716.0 ms | 0.90 ms | **796×** | 2 | `T.serial → T.reduce` + native fp16 I/O + `T.rsqrt` |
| argmax | TileLang | x:(8192,32768) dim=1 | 16.2 ms | 1.75 ms | **9.26×** | 13 | tiled shared-mem loads + `bM=256` + native fp16 I/O |
| matmul | Triton | A,B:(4096,4096) fp16 | 2.71 ms | 1.63 ms | **1.66×** | 6 | `@triton.autotune` + `GROUP_SIZE_M` L2 swizzle |
| conv2d | Triton | x:(32,256,128,128) | 32.1 ms | 12.5 ms | **2.57×** | 15 | implicit GEMM + fp16 tensor cores + padded input + autotune |

**Example trajectory (layer_norm TileLang, abridged from `layer_norm_iterations.md`):**

| iter | change | runtime | status |
|---|---|--:|---|
| baseline | `T.serial` reduction | 1090.0 ms | correct |
| 1 | replace `T.serial` with `T.reduce` | 5.24 ms | correct |
| 6 | float16 I/O + float32 accum | 3.02 ms | correct |
| 14 | native bf16 I/O, no conversion | 0.893 ms | correct |
| 15 | `out_idx` for output allocation | **0.891 ms** | correct (best) |
| 18 | 128 threads | 0.895 ms | correct |

---

### 9. Clock locking + measurement significance ⭐ *new this round*

**Evidence:** [`significance.csv`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/significance.csv) · [`clock_lock.txt`](experiments/results/NVIDIA_RTX_4000_Ada_Generation/clock_lock.txt) · script [`exp_significance.py`](experiments/exp_significance.py)

**Lock state.** Graphics 1410 MHz **held flat** across idle → sustained load (no thermal decay vs the unlocked 1545 → 1425 drift); memory pins to 8551 MHz under the 130 W cap, deterministic. Power 125–129 W < 130 W cap.

**Near-parity re-measurement (9 kernels × 2 impls, 100 reps each, locked clocks)**

| kernel | shape | impl | median ms | std ms | p95 ms | **E_lib %** | ci95 band | verdict |
|---|---|---|--:|--:|--:|--:|--:|---|
| layer_norm | (8192,8192) bf16 | triton | 0.920 | 0.006 | 0.930 | **94.46** | 1.43 | significant: DSL slower |
| softmax | (4096,32768) fp16 | triton | 1.810 | 0.006 | 1.822 | **95.18** | 0.76 | significant: DSL slower |
| mean_reduction | (8192,32768) fp32 | triton | 3.230 | 0.002 | 3.234 | **99.43** | 0.15 | significant: DSL slower |
| relu | (16384,16384) fp16 | triton | 3.425 | 0.004 | 3.430 | **104.04** | 0.52 | significant: **DSL faster** |
| add | (64M,) fp16 | triton | 1.282 | 0.003 | 1.285 | **101.21** | 0.84 | significant: **DSL faster** |
| swiglu | (4096,32768) fp16 | triton | 1.293 | 0.005 | 1.303 | **268.45** | 2.15 | significant: **DSL faster** |
| index_select | (65536,2048) fp16 | tilelang | 0.136 | 0.001 | 0.138 | **90.98** | 2.03 | significant: DSL slower |
| cross_entropy† | (4096,32768) fp32 | triton | 1.643 | 0.009 | 1.648 | **1455.6** | 15.73 | significant: **DSL faster** |
| conv2d | (32,256,128,128) fp16 | triton | 39.214 | 0.088 | 39.355 | **34.15** | 0.15 | significant: DSL slower |

† cross_entropy is implemented as **flash-CE** (fused softmax + NLL), so `F.cross_entropy` is not an equivalent baseline — the >14× ratio reflects the algorithmic gap, not pure DSL speed.

**Run-to-run rel-std = 0.0 – 0.9%** (vs the paper's 9% cross-session figure in Table 7). Every small efficiency gap resolves as statistically real — the revision **deletes the "9% clock variation" footnote** outright.

---

### 10. Minor corrections

**Evidence:** Verified against code (22 kernel directories under `ViperBench/`).

| issue | fix |
|---|---|
| kernel count says 21 in §1, but the artifact has 22 | **21 → 22** |
| "anamoly" typo | **"anomaly"** |
| Table 1 notation undefined | `16384²` = square fp16 GEMM · `64×128²` = batched, batch 64 of 128×128 |

---

### 11. ⏳ Open: Cross-architecture generality (A100 / H100)

**Runbook:** [`experiments/A100_H100_RUNBOOK.md`](experiments/A100_H100_RUNBOOK.md) — access secured.

The plan is to re-run the counter / conv-filter / FP32 / significance experiments on A100 and H100 and report whether the root causes and their relative impact survive beyond Ada (sm_89). This is the one genuinely new data-collection item; everything else above is in hand.

---

## Repository Layout

```
ASE-GPUDSL-ARTIFACT/
├── REBUTTAL.md                                  # First-class rebuttal artifact (§1–§4 + Appendix A)
├── REVISION_TODO.md                             # Forward-looking revision actions
├── reviews.txt                                  # Reviewer comments (line numbers cited throughout)
├── ase26-paper4134.pdf                          # Submitted paper
├── experiments/
│   ├── exp_*.py                                 # One script per rebuttal experiment
│   ├── A100_H100_RUNBOOK.md                     # Cross-arch runbook (open item)
│   └── results/NVIDIA_RTX_4000_Ada_Generation/  # All measured evidence above
├── ViperBench/                                  # 22-kernel PyTorch / Triton / TileLang benchmark
│   └── results/profile.csv                      # Per-kernel latency + memory
├── AKO4ALL/                                     # Iterative kernel-optimization loop (protocol + logs)
│   └── results/
│       ├── optimization_results.csv             # Per-campaign best speedups
│       └── optimized/<kernel>_iterations.md     # Per-iteration logs
└── logs/                                        # Full campaign history / working docs
```



---

## Environment

RTX 4000 Ada (sm_89) · torch 2.8.0+cu126 · triton 3.4.0 · tilelang 0.1.6.post1 · ncu 2024.3.2.0 · CUDA 12.6
