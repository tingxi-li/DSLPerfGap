# Pre-Submission Consistency Audit — ASE 2026 #4134

**Date:** 2026-05-26
**Method:** 4 parallel general-purpose agents, one per cross-cut, plus direct verification of every 🔴/🟠 finding against the source files.

| Agent | Cross-cut checked |
|-------|-------------------|
| A | `reviews.txt` ↔ rebuttal text (REBUTTAL_GAME_PLAN.md) |
| B | experiment data (`profile.csv`, NCU CSVs, NCU_FINDINGS.md) ↔ rebuttal/plan numbers |
| C | paper draft (`ase26-paper4134.pdf`) ↔ rebuttal claims & corrected mechanisms |
| D | paper draft ↔ codebase (`ViperBench/`, `AKO4ALL/`) |

**Severity:** 🔴 must-fix (reviewer-disprovable / blocks submission) · 🟠 should-fix · 🟡 minor.

---

## Verdict

**The rebuttal is sound and internally consistent.** Every correction it makes about the
paper is accurate (Agent C verified each against the PDF), every empirical claim is backed
by collected data to the digit (Agent B), and — critically — it **correctly avoids
re-asserting the three mechanisms our own counters refuted** (RC0a sync, RC3 conv-spilling,
RC4 Winograd-primary). Its "ground and sharpen" framing is not just defensible; it is
*necessary*, because the **paper itself explicitly asserts all three** (Agent C), and a
reviewer with our counter data could disprove them.

The findings below split cleanly into two buckets:
- **§1 — rebuttal/doc fixes** (this round): small, mostly already applied this session.
- **§2 — paper-text & code fixes** (revision round): author decisions; the most important
  is that the paper makes claims the artifact does not yet reproduce (N1) and asserts
  mechanisms our data overturns (RC0a/RC3/RC4).

---

## §1 — Rebuttal & doc-hygiene (this round)

### 🔴 1.1 — "R2-Q4" mislabel on the Nsight-counters paragraph  *(Agent A — FIXED)*
REBUTTAL_GAME_PLAN.md:74 tagged the counters paragraph `(R1-Q4, R2-Q4)`. Verified against
`reviews.txt:148`: **R2-Q4 is "mitigation experiments for RC2b, RC3, and RC4," not counters**
(the counter ask is R1-Q4 only; reviews.txt:85). The paragraph *does* contain the requested
isolations (L2-residency→RC2b, register/spill→RC3, Winograd→RC4), so the fix is to name them
explicitly so the `R2-Q4` tag is earned rather than incidental.
**Applied:** heading renamed to cover both asks; last sentence now maps each isolation to its
RC. The `(R1-Q4, R2-Q4)` tag is now accurate.

### 🟠 1.2 — `cross_entropy` efficiency stated two ways  *(Agent B — FIXED)*
ADDITIONAL_EXPERIMENTS_PLAN.md:64,66 said **Triton 1452% / TileLang 88%** (from the fresh
isolated `ce_timing` harness, Triton 1.64 ms). The per-kernel element-wise **table is sourced
from `profile.csv`**, which gives **Triton 1277% / TileLang 86%** (Triton 1.8684 ms — the
original sweep's row, not re-measured). Both show the same robust qualitative result (Triton
≫100%, TileLang <100%); only the % differs by measurement provenance.
**Applied:** standardized the docs on the **profile.csv-derived 1277% / 86%** (the table
source), with a parenthetical that a fresh isolated run corroborates (~1450%).

### 🟡 1.3 — Stale "regenerate profile.csv (15908 ms)" instructions  *(Agent B — FIXED)*
profile.csv was **already surgically patched** this session (cross_entropy pytorch rows now
0.2168 ms small / 23.8647 ms large; E_lib 742%/238% small, 1277%/86% large). But four docs
still instructed "regenerate profile.csv, it holds the old 15908 ms loop rows":
REBUTTAL_GAME_PLAN.md:41,112 · ADDITIONAL_EXPERIMENTS_PLAN.md:167 · REVIEWER_WEAKNESS_ANALYSIS.md:152.
**Applied:** all four now read "patched (done)" with the current figures.

### 🟡 1.4 — "851,000%" provenance  *(Agent B — noted)*
The "~851,000%" magnitude came from the *original* profile.csv (pytorch 15908 ms ÷ triton
1.8684 ms ≈ 851,000% — actually 851,000% ≈ 15908/0.00187…; the loop row drove it). A separate
contaminated harness (`fused_baselines.csv`, eager row 4015 ms) reads ~245,686%. Both are
contamination artifacts of the old loop; neither is reproducible now. Docs retain "~851,000%"
as the headline illustration with the explanation; this is fine — flagged only so we don't
later cite the two magnitudes as if measuring the same thing. **No action needed.**

### 🟠 1.5 — R1's "is there an in-DSL / compiler-level fix?" (W4) not answered  *(Agent A — FLAG)*
reviews.txt:31 (R1, W4) asks whether the gaps are fixable in-DSL or only upstream. The rebuttal
implies it (RC0a "correctable in user space," the T.reduce path) but never states it as the
answer to W4. **Recommended one-liner** (if budget allows): *"Two of the gaps are user-space
fixable today (the T.reduce/LDG rewrites, which our mitigation kernels demonstrate); the rest
require upstream compiler changes (TF32 lowering, autotune search) — we will state this fix
taxonomy explicitly."* → author's call given the 750-word budget.

### 🟡 1.6 — W13 (A100→Ada RC3 label slip) and R3 ("calibrate claims to scope") not surfaced  *(Agent A — FLAG)*
Minor reviewer points not addressed in 750 words. Likely fine to fold into the revision letter
rather than the rebuttal. Author's call.

---

## §2 — Paper-text & code (REVISION round — author decisions)

> Tracked as a checkable list in **`REVISION_TODO.md`**.

These are **not** rebuttal-round edits (the rebuttal correctly handles them by promising
revision). They are listed so nothing is lost when the revision is written. Items 2.1–2.3 are
the ones a reviewer **with our data could actively disprove**, so the paper text *must* change.

### 🔴 2.1 — Paper asserts the three refuted mechanisms  *(Agent C — paper-text)*
Our collected counters overturn three mechanism claims the paper makes **explicitly**:

| RC | Paper says (location) | Our data says |
|----|------------------------|---------------|
| **RC0a** | "sequential thread synchronization"; Table 5 counter "Thread sync count, warp stall"; "log₂(256/32)=3 barriers" (§6, p5 L527-541) | **memory-latency-bound**: `barrier ≈ 0`, `long_scoreboard` dominates the stall breakdown |
| **RC3** | conv K≥5 "register pressure / spill to local memory"; Table 5 row "Conv2d (K≥5)"; `sm__register_spill` (abstract; §6 p6 L688-702) | **conv n_spills = 0**; the 51 GB spill is in **TileLang layer_norm** (254 regs, occ 16.5%) — RC3 is real but **mis-attributed to the wrong kernel** |
| **RC4** | "remaining gap **primarily** associated with missing Winograd" (abstract); "remaining 20% deficit attributed to absent Winograd" (§7.4) | Winograd isolation shows **~2–3%**, not "primary" |

**Action (revision):** edit abstract, §6, §7.4, **and Table 5** so the mechanism labels match
the measured counters. The rebuttal already promises exactly this ("where they sharpen an
attribution… we will update the analysis"). The corrected story is *stronger* (it is now
counter-grounded), but the text as written is disprovable and must move.

### 🔴 2.2 — N1: artifact does not reproduce the paper's stated conditions  *(Agent D — code/paper)*
Verified by grep across `ViperBench/`:

| Paper claims | Code reality |
|--------------|--------------|
| "NHWC layout" (§3.2 L281-282; Table 2 caption L581) | **no** `channels_last`/`memory_format`/NHWC anywhere; conv2d does `input.contiguous()` → **NCHW** (conv2d/triton_impl.py:111) |
| "allow_tf32=False" (§3.2 L281-282) | **no** global `torch.backends.cuda.matmul.allow_tf32` set (the `allow_tf32` hits are per-kernel Triton `tl.dot` flags — a different mechanism) |
| "cudnn.benchmark=False" (§3.5 L341-342) | **no** `torch.backends.cudnn.benchmark` set anywhere |

This is entangled with **RC1**: the paper's RC1 narrative is "NHWC breaks LDG.128 alignment"
(p6 L614-617), but the measured conv ran **NCHW**. A reviewer grepping the artifact finds none
of these flags → reproducibility gap that undercuts RC1.
**Action (revision):** *either* (a) add the flags + `channels_last` and re-validate results,
*or* (b) soften §3.2/§3.5/Table-2 to describe the actual NCHW + default-flags setup and
re-examine whether the RC1 "NHWC" mechanism wording still holds. **Author decision — affects
results; do not change benchmark code unilaterally.**

### 🟠 2.3 — Timing methodology mismatch  *(Agent D — paper-text)*
Paper §3.3 L297-301: "CUDA events (`cudaEventRecord`/`cudaEventElapsedTime`)… report the
**average**." Code (`benchmark.py:41-48`): `time.perf_counter()` + `torch.cuda.synchronize()`,
reports the **median** (`times.sort(); times[len//2]`). 10 warmup / 100 measured and
peak-memory are correct. **Action:** align the text to the code (perf_counter + median) — the
simplest fix, no re-measurement needed.

### 🟠 2.4 — `E_lib` defined but never computed by committed code  *(Agent D)*
Paper §3.4 defines `E_lib = t_library / t_DSL × 100%`, but no committed script emits it;
`profile.csv` carries raw latencies and the table percentages are hand-derived. **Action:**
add a tiny script that derives E_lib from profile.csv (cheap; also de-risks transcription
errors) — recommended for the artifact even if the text stays.

### 🟡 2.5 — "21 kernels" stray  *(Agents C & D)*
Paper says 22 everywhere except §2.5 p2 L137-138 ("21 kernels"). The code has 22 dirs. The
rebuttal's "21→22" correction is valid. **Action:** fix the one stray in the revision.

---

## §3 — Verified clean (no action)

- **Reviewer identities, scores, weaknesses W1–W13, attributions** — faithful (Agent A).
- **Full 24-row Nsight counter table** — NCU_FINDINGS.md matches `ncu_summary.csv` to the digit
  (Agent B): layer_norm tilelang 254 regs / 51.5 GB spill-load / occ 16.5% / barrier 0 /
  long_scoreboard 104.91 / DRAM 90.2%; conv2d triton 128 regs / 0 spills / load-eff 36.4%;
  cuBLAS 99.7% load-eff; argmax L2 hit 0.64%.
- **Headline gap numbers** — 31.7%/56.4% GEMM, 314× LayerNorm, 2067×/99.6% FP32, fused
  F.layer_norm 94.6%, §5 Δ=0pp / §7.3 1.66× — all reproduced; only mechanism *labels* change
  (Agents B, C).
- **Rebuttal's paper-fact corrections** — F.layer_norm/94.6% (§3.2 L283, Table 3), "anamoly"
  typo (abstract L25), Table 1 notation, RC0(a)/(b) split mirrors paper structure (Agent C).
- **Codebase ↔ paper structural facts** — 22 kernels, tolerances (test_utils.py:12-25), input
  shapes, AKO4ALL numbers (1224× / 1.66× / 2.57×), unified-API contract (Agent D).
- **No new research numbers in the rebuttal** — integrity gate holds (Agent B).

---

## §4 — Prioritized actions

**This round (rebuttal):**
1. ✅ 1.1 R2-Q4 mislabel — FIXED.
2. ✅ 1.2 cross_entropy 1452→1277% reconciled to profile.csv — FIXED.
3. ✅ 1.3 stale "regenerate profile.csv" ×4 — FIXED.
4. ⏳ 1.5 W4 in-DSL/compiler-fix sentence — **author decision** (word budget).
5. ⏳ 1.6 W13 / R3 — fold into revision letter (author).

**Revision round (flagged, author decisions):**
6. 🔴 2.1 RC0a/RC3/RC4 mechanism text + Table 5 — mandatory (reviewer-disprovable).
7. 🔴 2.2 N1 NHWC/allow_tf32/cudnn.benchmark — add flags & re-validate, or soften text.
8. 🟠 2.3 timing text (CUDA-events/mean → perf_counter/median).
9. 🟠 2.4 ship an E_lib script.
10. 🟡 2.5 "21 kernels" stray.
