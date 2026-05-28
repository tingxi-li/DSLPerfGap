# What Reviewers Asked For on Profiling / Benchmarking Rigor — ASE 2026 #4134

**Date:** 2026-05-26 · **Method:** 3 parallel agents — reviews.txt (asks), paper PDF (claims), codebase (implementation).
**Reviewer key:** R1 = #4134A (Weak Reject, the rigor gatekeeper + only Artifact/DAS reviewer), R2 = #4134B (Weak Reject), R3 = #4134C (Weak Accept).

> **One-line verdict:** the reviewers' single most pointed rigor critique — *"clocks may not be locked, so 94.6% vs 97.8% may be meaningless"* (R1) — is **unaddressed in both the paper (silent) and the code (no locking anywhere)**. That is precisely the clock-lock + warmup work now in progress. The second-biggest ask — *show the measured Nsight counters* — is **already answered** by the completed counter run.

---

## 1. The reviewer asks (by rigor category)

### 🔴 A. Clock / thermal / power stability + significance of small deltas — **R1, the crown-jewel critique**
- **R1, reviews.txt:50:** *"Table 7 notes a 9% variation in baseline performance across profiling runs for the same convolution setup, attributed to GPU clock fluctuations. It is not stated whether clocks were locked. Without that, small efficiency differences (e.g., 94.6% vs. 97.8%) may not be meaningful."*
- What they want: lock clocks, **or** report variance/repeats so sub-10% efficiency gaps are defensible.
- This is the **only pure timing-noise objection** in all three reviews, and the most quotable single line. Category: clock-thermal-power + statistical-significance.

### 🔴 B. Show the *measured* hardware counters (don't just describe the methodology) — **R1 + R2**
- **R1, reviews.txt:62:** *"Section 3.3 introduces detailed Nsight Compute profiling, but the paper does not show the corresponding counter data. This makes it difficult to assess or verify the analysis."*
- **R1, reviews.txt:85 (Q4):** wants specific counters — *"vectorized load utilization for convolution vs. GEMM, register spill indicators for RC3, and warp stall breakdowns for RC0."*
- **R1, reviews.txt:30:** RC1/RC2 backed by data+mitigation; **RC3 lacks profiling evidence**, **RC4 inferred by elimination** not controlled isolation.
- **R1, reviews.txt:58:** RC3 register-spill/occupancy for 5×5+ filters *"are not actually benchmarked, and no supporting data is shown… reads more like a hypothesis."*
- **R2, reviews.txt:131/148 (Q4):** RC2b/RC3/RC4 have *"no direct mitigation experiment"* — provide them or justify infeasibility.
- Category: profiling-counters. **Two of three reviewers.**

### 🟠 C. Correctness/tolerance as a *precondition* for valid latency comparison — **R3 (protects the one Accept vote)**
- **R3, reviews.txt:174/180/186:** wants documented tolerances, failure counts, input distributions, edge cases, and **re-validation of mitigation kernels** — latency gaps only matter if kernels are semantically equivalent.

### 🟠 D. Baseline-provenance asymmetry — **R1 + R3 consensus**
- **R1, reviews.txt:29** + **R3, reviews.txt:182:** conv vs cuDNN-with-Winograd but normalization/element-wise vs **unfused eager PyTorch** → the single "library efficiency" measures different things per category; *"Overall ∼65%" potentially misleading.* Want split vendor / eager / fused metrics.

### 🟠 E. Cross-architecture generality of the measurements — **R1 + R3**
- **R1, reviews.txt:32/64/87** + **R3, reviews.txt:178/187:** all data on one sm_89 GPU, yet **RC3 cites A100 register specs** (R1:64). Do magnitudes hold on A100/H100?

### 🟡 F. Tuning-methodology provenance — **R1 + R2**
- **R1, reviews.txt:48** + **R2, reviews.txt:133:** the Δ=0pp (§5) vs 1.66× (§7.3) discrepancy signals an under-specified measurement; specify parameters + search space.

---

## 2. Asked vs. Claimed vs. Implemented (the actionable gaps)

| Rigor dimension | Reviewer ask | Paper claims (Agent B) | Code does (Agent C) | Status |
|---|---|---|---|---|
| **Clock locking** | R1:50 — lock or it's noise | **SILENT**; Table 7 footnote *admits* 9% drift "attributable to run-to-run GPU clock variation" (⇒ clocks **not** locked) | Clocks now locked (graphics 1410 / memory 9001 MHz, verified flat under load in `clock_lock.txt`); `experiments/exp_significance.py` self-checks the lock before timing | ✅ **DONE** — locked re-measurement: run-to-run rel-std **0.0–0.9%** (vs the 9%); see `significance.csv` |
| **Dispersion / error bars** | R1:50 — are small deltas meaningful? | **SILENT** for latency; only "stability within 2% across five runs" for *counters* | `experiments/exp_significance.py` re-measured the near-parity set via `_harness.time_kernel` and emits median+std+p95 + a propagated 95% band + significance verdict | ✅ **DONE** — all near-parity gaps resolve as **statistically real** (e.g. layer_norm 94.5%±1.4pp, softmax 95.2%±0.8pp); none "within noise" |
| **Timing primitive / statistic** | (rigor landmine under R1) | "**CUDA events** (cudaEventRecord/ElapsedTime)… report the **average**" (§3.3) — but Table 4 caption says "**Median**" | `benchmark.py` (the path that built `profile.csv`) = **perf_counter + median**; `_harness.py` = CUDA-events + median+mean+std; `bench.py` = CUDA-events + **mean** | 🟠 paper↔code mismatch + paper-internal mean/median mismatch — a reviewer grepping the artifact finds a different method |
| **Measured counters** | R1:62/85, R2:131 | Names ncu + counter groups; Table 5 lists per-RC counters but **no numeric values** (shows latency-recovery instead) | `ncu_counters.sh` collects exactly the asked sets (loads/regs/stalls/L2-DRAM); `consolidate_ncu.py` summarizes | ✅ **answered** — counter run completed (24/24 → `NCU_FINDINGS.md`). *(Agent C saw the script's `RmProfilingAdminOnly` guard — that block is stale; it ran post-reboot.)* |
| **Setup flags** | (validity precondition) | "`cudnn.benchmark=False`" (§3.5); conv "**NHWC** + `allow_tf32=False`" (§3.2) | Production benchmarks set **none** of these; conv runs **NCHW**. Flags appear only in rebuttal diagnostics (`exp_fp32_gemm.py`, `exp_winograd_isolation.py`) | 🔴 N1 (see `REVISION_TODO.md`) |
| **Correctness/tolerances** | R3:174/180/186 | "limited detail" (R3's complaint) | `test_utils.py` per-dtype tolerances + `loose_tol`; edge + mitigation-revalidation experiments exist | 🟢 addressable from artifact + completed work |

---

## 3. Bottom line

**Acceptance hinges on R1, and R1's two load-bearing rigor asks are #A (clocks/significance) and #B (show counters) — both now done.**
- **#B done** — the completed Nsight run delivers exactly the counters R1 enumerated (and corrected RC0a/RC3/RC4).
- **#A done** — clocks locked (graphics 1410 / memory 9001 MHz, verified flat under the 130 W cap in `clock_lock.txt`) and the near-parity comparisons re-measured with dispersion (`experiments/exp_significance.py` → `significance.csv`): locked run-to-run variation is **0.0–0.9%** (vs the paper's 9%), and every small efficiency gap R1 worried about resolves as **statistically real, not noise**. Rebuttal `L72` now reflects this; the locked numbers are revision material (rebuttal text stays qualitative). This lets the revision *delete* the Table 7 "9% clock variation" footnote.
- **Revision follow-ups** (tracked in `REVISION_TODO.md`): decide whether to re-baseline the paper's tables at locked clocks (some locked E_lib differ from `profile.csv` — e.g. softmax 95.2% vs 97.6%), fix the timing-method mismatch (paper: CUDA-events+mean; code: perf_counter+median), and close the N1 flag gap.

Net: of the six rigor asks, one is already satisfied (counters), one needs the in-progress clock/warmup/dispersion work (the decisive one), and the rest are paper-text/baseline-split/cross-arch items tracked for the revision.
