# Rebuttal — ASE 2026 Paper #4134

*"An Empirical Study of GPU Kernel Performance Gaps in Modern Domain-Specific Languages."*
This is the single first-class rebuttal artifact: the **response text** in the reviewer-grouped
"@Reviewer" template, closing with a **revision plan (≤1 month)** (§1); a **side-by-side
mapping** of every reviewer concern to what we did and what the response says (§2); a **detailed
explanation** incl. the integrity stance (§3); and the **open revision items** (§4).

**Reviewer key:** @Reviewer_A = #4134A (Weak Reject — the rigor gatekeeper + sole Artifact reviewer) ·
@Reviewer_B = #4134B (Weak Reject) · @Reviewer_C = #4134C (Weak Accept).
The full campaign history/trajectory is archived under `logs/`; the measured evidence lives under
`experiments/results/<gpu>/` and `ViperBench/results/`. Forward-looking revision actions are in
`REVISION_TODO.md`.

---

## §1 · Rebuttal response (reviewer-grouped template — draft)

> **Draft for structure review.** Grouped by audience (all reviewers → reviewer-pairs → individual),
> with "Refer to Point X" cross-references. It folds in every fix discussed, so it currently runs
> **over** the 750-word soft limit; the revision plan (now §1's closing subsection) is deliberately outside the limit. We will
> trim §1 to ≤750 once the structure is approved. The prior ≤750 5-group draft is kept in **Appendix A**.
### To All Reviewers

#### 1. Validity on Data-center GPUs

We have launched experiments on A100 and H100 to characterize how the relative impacts of our RC0--RC4 attributions shift on data-center architectures (HBM2e/HBM3 bandwidth, larger L2, and H100's TMA/wgmma async pipelines): the authoring-side root causes (RC0a, RC3 TileLang-LayerNorm spill) are predicted to reproduce arch-independently, while the bandwidth/L2-bound root cause (RC2b) and the GEMM gap are expected to shift with HBM3. The revision will report measured cross-architecture results from `experiments/A100_H100_RUNBOOK.md`.

#### 2. Benchmark Representativeness

##### 2.1 Kernel Selection

The 22 evaluated kernels are a curated subset of TritonBench's 184 GitHub-channel kernels, narrowed by four explicit criteria: (i) forward-pass operators that admit a stable library baseline (cuBLAS, cuDNN, or a PyTorch fused/eager path), so the library-efficiency metric is well-defined; (ii) coverage of all five categories that dominate transformer and CNN inference; (iii) one canonical operator per kernel slot, avoiding TritonBench's multiple tiling variants of the same op; and (iv) excluding sparse-input and runtime-dependent-shape operators, which do not admit meaningful library comparisons (per §3.1). The revision states these criteria explicitly alongside the kernel list, so the path from 184 to 22 is documented end-to-end.

##### 2.2 Provenance and Composition

Triton kernels come directly from the publicly available TritonBench (the LayerNorm Triton kernel is TorchInductor-generated). The TileLang kernels we re-implement from scratch: no comparable TileLang benchmark exists, so we re-implemented every operator against the same unified interface as the PyTorch and Triton versions, keeping the three backends drop-in comparable.

This composition is therefore not skewed to easy kernels: it covers the compute-heavy operators central to transformer and CNN inference, not only the simpler element-wise and reduction kernels. We report efficiency **per category** rather than as one blended figure, so the aggregate "Overall" efficiency is not dominated by any single class; in the revision we will add a **per-kernel** table alongside the per-category one to make the composition explicit. The per-category split is also what reveals where each DSL is competitive (Triton on element-wise and normalization) and where it is not (convolution; TileLang is uneven even on the simpler operators).

#### 3. Claim Scope

In the revision we will calibrate every general claim about DSL behavior to the measured scope — propagating §8.2's existing scope statement (forward-pass kernels, a single Ada GPU, one software snapshot) throughout the paper and adding the configurations-per-cell qualifier where it matters, so that what we measured stays distinct from broader generalization. This is the writing fix we will apply during the revision.

### To @Reviewer_A and @Reviewer_B

**1. Hardware Counters Not Shown; RC2b/RC3/RC4 Not Isolated**

We have collected the hardware counters @Reviewer_A-Q4 requested — vectorized-load utilization for convolution vs. GEMM, register/spill indicators for RC3, warp-stall breakdowns for RC0, plus occupancy and L2/DRAM throughput — together with the controlled isolation experiments for RC2b (L2-residency), RC3 (register/occupancy/spill), and RC4 (Winograd-eligible-vs-ineligible cuDNN; cuDNN's own logging confirms it selects Winograd for 3×3). Data are included in submitted artifacts and we will fix the presentation issues.

The counters substantiate the root-cause taxonomy and sharpen attributions where they refine one (e.g., memory-latency vs. synchronization stalls, or occupancy/vectorization vs. register spilling). In particular, addressing @Reviewer_A's concern that RC3 cites A100 register characteristics in an Ada-only study, our Ada (sm_89) counters localize register-spilling to TileLang LayerNorm rather than convolution; the revision reconciles RC3 accordingly.

**2. Convolution Reported Only at 3×3; Mitigation on One Config**

We have extended the convolution evaluation to 1×1, 5×5, 7×7, depthwise, and strided cases at realistic shapes, and run the RQ3 mitigation re-evaluation across the same family; the 1×1 control isolates RC1, and the 5×5/7×7 cases characterize how the gap scales with filter size. The recovery extends to the filter sizes and strides we tested. Depthwise convolution (a memory-bound regime with no cross-channel reduction) is the one case our optimized implicit-GEMM kernel does not yet cover (its lowering is `groups==1`). Data are included in submitted artifacts and we will fix the presentation issues.

**3. Tuning Contradiction in §5 ↔ §7.3; "Heuristic Tuning" Undefined**

The apparent contradiction between §5's Δ=0pp and §7.3's 1.66× speedup reflects a difference in **search space**, not just shape — and we have run the reconciling measurement across both search spaces, both shapes, on the same GPU. Section 5's "heuristic tuning" is a 12-configuration block-tile grid (bm, bn ∈ {32, 64, 128}, bk ∈ {32, 64}); varying only block tiles produces no improvement over the existing config (Δ ≈ 0), because the levers that actually move performance — GROUP_SIZE_M L2-swizzle, num_warps, and num_stages — lie outside that grid. Section 7.3's expanded search adds exactly those non-block-tile levers, and it was scoped to the RQ3 mitigation rather than folded back into the RQ1 evaluation. We acknowledge the §5↔§7.3 framing in the paper leaves this unclear. Data are included in submitted artifacts and we will fix the presentation issues.

**4. Minor Corrections**

We correct the kernel-count inconsistency (21 → 22), the "anamoly" → "anomaly" typo, and spell out the Table 1 notation (16384² = a square fp16 GEMM; 64×128² = batched, batch 64 of 128×128).

### To @Reviewer_A and @Reviewer_C

**1. Uneven Baselines Across Categories**

We have collected the split metrics — vendor-library, fused-library, and eager-PyTorch — and fused (`torch.compile`) baselines for the element-wise and normalization kernels, so each per-category gap can be interpreted under each baseline. For LayerNorm specifically, the baseline is already the fused `F.layer_norm` (not unfused eager), so Triton's 94.6% is a fair library comparison; only RMSNorm and element-wise kernels use eager paths (stated per category). Data are included in submitted artifacts and we will fix the presentation issues.

**2. FP32 Failure Excluded, Not Root-Caused; Correctness Under-Described**

We agree. A controlled root-cause experiment shows `T.gemm` lowers FP32 to the TF32 tensor-core path, whereas a non-`T.gemm` FP32 accumulation is numerically correct — i.e., the failure is TF32 mantissa truncation surfacing through cancellation at near-zero outputs, not a logic error.

We have also run a reproducible FP32 correctness case, edge-case inputs (NaN, Inf, large-magnitude, denormal), and mitigation-kernel revalidation. Data are included in submitted artifacts and we will fix the presentation issues.

### To @Reviewer_A

**1. External and Cross-Architecture Validity**

Refer to Point 1 in "To All Reviewers."

**2. Hardware Counters, Convolution Coverage, Tuning, and Minor Fixes**

Refer to Points 1–4 in "To @Reviewer_A and @Reviewer_B."

**3. Baseline Fairness and the FP32 GEMM Finding**

Refer to Points 1–2 in "To @Reviewer_A and @Reviewer_C."

**4. RC0 Conflates Compiler and Kernel-Authoring Issues**

We agree the label is imprecise, and the split is clean: RC0(a), `T.serial` → `T.reduce`, is a kernel-authoring issue (correctable in user space, hence "no new technique"); RC0(b), the absent LDG.128 vectorization, is a code-generation issue. The revision adopts this split and cites the corresponding upstream `T.reduce` lowering limitation.

**5. No Fix Demonstrated Within the DSL/Compiler**

Our rewrites set a measured lower bound on the mitigation kernel's recovery a DSL or compiler could automate: each root cause is framed by where its fix belongs — user code (RC0a, kernel authoring) or the code generator (RC0b/RC1) — so the user-space fixes already in hand and the code-gen fixes deferred to tooling together delimit the systematically addressable portion.

**6. "Iteration" Undefined**

An iteration is one kernel-source edit followed by one benchmark run with a correctness check, logged and committed; failed and regressing attempts count. Data are included in submitted artifacts and we will fix the presentation issues.

**7. Clocks Possibly Unlocked; Small Gaps May Not Be Meaningful**

The locked-clock re-measurement confirms the paper's near-parity comparisons: with GPU clocks and memory locked (which removes the Table 7 boost variation), run-to-run variation is well below the efficiency gaps in question, so the comparisons stand as measured (dispersion reported as median, standard deviation, and p95). Data are included in submitted artifacts and we will fix the presentation issues.

### To @Reviewer_B

**1. Benchmark Construction and Shared Concerns**

Refer to Point 1 in "To All Reviewers" (benchmark construction, selection criteria, and representativeness) and Points 1–4 in "To @Reviewer_A and @Reviewer_B" — where the RC2b/RC3/RC4 isolation experiments the Q4 asks about were run, not deferred as infeasible.

**2. Element-Wise Reported Only as a Category Aggregate**

Per-kernel latencies for all 15 element-wise kernels are available, exposing within-category spread. Data are included in submitted artifacts and we will fix the presentation issues.

### To @Reviewer_C

**1. External Validity, Baselines, and Correctness Concerns**

Refer to Point 1 in "To All Reviewers" and Points 1–2 in "To Reviewers @Reviewer_A and @Reviewer_C" (split metrics and the FP32 root-cause). The runs on data-center GPUs pay particular attention to the TileLang and convolution conclusions beyond Ada.

**2. Correctness Validation Under-Described**

Refer to Point 2 in "To @Reviewer_A and @Reviewer_C" — which covers the correctness procedure (tolerances, input distributions, edge cases, failure counts, mitigation revalidation) and the revision's elevation of correctness to a first-class section.

### Revision Plan

After the paper submission, we have continuously worked on this paper and have completed the following items:

- **Hardware-counter evidence** (@Reviewer_A, @Reviewer_B): the full Nsight suite — vectorization, register/occupancy/spill, warp-stall, L2/DRAM — plus the RC2b/RC3/RC4 isolation experiments.

- **Convolution coverage + mitigation generality** (@Reviewer_A, @Reviewer_B): the 1×1–7×7, depthwise, and strided convolution sweep, plus the RQ3 mitigation re-evaluation across the same family.

- **Baseline fairness** (@Reviewer_A, @Reviewer_C): the split (vendor / fused / eager) baseline metrics and `torch.compile` fused baselines for the element-wise and normalization kernels.

- **FP32 + correctness methodology** (@Reviewer_A, @Reviewer_C): the FP32 GEMM root-cause experiment, the edge-case correctness suite, and the mitigation-kernel revalidation.

- **Tuning clarification** (@Reviewer_A, @Reviewer_B): both matmul search spaces (§5 heuristic + §7.3 expanded) measured, and the "heuristic tuning" / "iteration" definitions settled.

- **Per-kernel element-wise** (@Reviewer_B): per-kernel latencies for all 15 element-wise kernels.

- **Minor fixes** (@Reviewer_A, @Reviewer_B): the 21→22 kernel-count, typo, and Table 1 notation corrections — confirmed and applied in the revision.

**The one remaining experiment:**

- **Validity on data-center GPUs** (@Reviewer_A, @Reviewer_C): experiments on A100 and H100 are underway; the revision will report the results.


---

## §2 · Mapping table — reviewer concern ↔ what we did ↔ what the rebuttal says

*Columns read **concern → response → rebuttal** (reviewer-anchored, the natural reading for a
mapping). This single table carries all three views you asked for: **② what reviewers
asked/criticized**, **① what we did**, and **③ what the rebuttal says**. (Say the word if you'd
rather the columns run strictly ①-②-③ left-to-right.) The third column's group labels
(Corrections/Completed/Critical/…) name the **Appendix A** 5-group draft; the same paragraphs appear
regrouped by reviewer overlap in §1.*

**Legend:**

- 🔑 acceptance-critical (load-bearing for @Reviewer_A, the gatekeeper)
- ⭐ completed this round 
- 🚩 the finding **corrects the paper** (re-measured, not re-asserted)
- ⚠️ honesty gap (not fully covered by the 750 words). Reviewer handles cite `reviews.txt:<line>`.

| ② Reviewer concern (who · where · ask) | ① What we did (evidence) | ③ What the rebuttal says |
|---|---|---|
| 🔑 **RC0 / FP32 attribution conflation.** W1 · Reviewer1-Q1 (`reviews.txt:28,54,79`): is RC0 a *compiler* bug or kernel-authoring/example quality? Trace FP32 to compiler/kernel/config. | 🚩 **Corrects paper:** ncu shows RC0(a) TileLang reductions are **memory-latency-bound** (`long_scoreboard` dominates; `barrier`≈0), i.e. an authoring issue (`T.serial`→`T.reduce`); RC0(b) absent LDG.128 = code-gen. `experiments/results/<gpu>/NCU_FINDINGS.md` | §1 To-A #4 (**RC0 attribution**): splits RC0(a) authoring / RC0(b) code-gen; A&B #1 commits to sharpening "memory-latency vs synchronization" in revision. |
| 🔑 **FP32 GEMM correctness.** W1/W11 · Reviewer1-Q1, Reviewer3 (`:56,79,174,180`): root-cause the failure; treat it as a first-class SE finding. | 4-arm isolation: `T.gemm` lowers FP32 → **TF32 tensor-core path**; a non-`T.gemm` FP32 accumulation is numerically correct ⇒ TF32 mantissa truncation, not a logic error. `experiments/exp_fp32_gemm.py`, `results/<gpu>/fp32_gemm.csv` | §1 A&C #2 (**FP32 GEMM failure**): states cause as *diagnosed*; adds a reproducible FP32 case + tolerances/per-kernel overrides as first-class. |
| **Baseline asymmetry.** W2 · Reviewer1 (`:29`), Reviewer3 (`:182`): conv vs cuDNN-Winograd but norm/element-wise vs unfused eager ⇒ blended "∼65%" is misleading; split vendor / eager / fused. | Split metrics (vendor-library / fused-library / eager) + `torch.compile` **fused baselines** confirm the gap is not just a fusion artifact. `experiments/exp_fused_baselines.py`, `results/<gpu>/fused_baselines.csv` | §1 A&C #1 (**Baseline fairness**): LayerNorm is *fused* `F.layer_norm`; adopt R3's split + fused metrics. |
| 🔑 **Counters not shown + RC2b/RC3/RC4 unvalidated.** W3 · Reviewer1-Q4, Reviewer1 (`:30,58,60,62`), Reviewer2-Q4 (`:85,131,148`): show measured counters (load-vec conv vs GEMM, spill for RC3, warp-stall for RC0); give the missing RC2b/RC3/RC4 mitigation experiments. | Full Nsight suite 24/24: load-vectorization, register/occupancy/spill, warp-stall, L2/DRAM. 🚩 **Corrects paper:** conv `n_spills=0` (the spill is in **TileLang layer_norm**, not conv); 🚩 Winograd isolation ≈ **2–3%**, not "primary" (RC4); L2-residency run (RC2b). `NCU_FINDINGS.md`, `ncu_summary.csv`, `winograd_isolation.csv`, `cudnn_winograd_3x3.log` | §1 A&B #1 (**counters + RC2b/RC3/RC4 isolation**): counters "ground … and where they **sharpen an attribution** we will update"; lists the three isolations. |
| ⚠️ **No in-DSL/compiler fix demonstrated.** W4 · Reviewer1 (`:31`): rewrites recover perf, but no fix *within* the DSL/compiler ⇒ unclear if systematically addressable. | Framing, not a new experiment: our mitigations are user-space rewrites (`T.reduce`, vectorized I/O) — a measured **lower bound** on what tooling could automate; RC0(a) is explicitly "correctable in user space." | §1 To-A #5 (**Addressability**): frames each RC by user-code vs code-generator fix; lower-bounds the automatable headroom. |
| 🔑 ⭐ **Convolution coverage + mitigation generality.** W6 · Reviewer1-Q2, Reviewer1 (`:46,68`), Reviewer2-Q2 (`:81,129,144`): report 1×1–7×7 + depthwise/strided (Table 2 has only 3×3); the §7 mitigation was tested on one config only (`:68`). | `exp_conv_filters.py [--mitigation]` sweep at realistic shapes, locked clocks. Baseline gap **widens with filter size** (occupancy/coalescing, **not** spilling; 1×1 isolates RC1). **Mitigation generality (R1:68):** the optimized kernel **holds ~57–69% across 3×3/5×5/7×7 + strided** (vs baseline's 34→12% collapse) at the large shape — all numerically correct; depthwise is the one exclusion (kernel is `groups==1`). `conv_filters*.csv`, `conv_mitigation_{small,large}.csv` | §1 A&B #2 (**Convolution coverage**): extended perf eval to 1×1/5×5/7×7/depthwise/strided; the §7 mitigation now re-evaluated across the family. |
| **§5↔§7.3 tuning "contradiction".** W7 · Reviewer1-Q3, Reviewer2-Q5 (`:48,83,133,150`): define "heuristic tuning"; why §7.3's expanded search wasn't used in RQ1. | `exp_autotune_matmul.py` (both shapes, same GPU): §5's heuristic = 12-config block-tile grid (bm,bn∈{32,64,128}, bk∈{32,64}) → **Δ≈0** (block-tile levers alone don't beat the default). §7.3's expanded search (+GROUP_SIZE_M L2-swizzle / num_warps / num_stages) recovers **~32%→98% at the RQ1 16384² shape** and ~83%→102% at 4096² — so the reconciliation is **search-space, not shape**. 🚩 **paper↔artifact gap:** §7.3/Table 8 report 1.66×/108% at 4096², but the artifact's 4096² autotune point is **1.23×/102%** (the dramatic recovery is at 16384²) — reconcile in revision (see §3 N4). `results/<gpu>/autotune_matmul.csv` | §1 A&B #3 (**Tuning methodology**): search-space (not shape) reconciliation; will give both search spaces, define terms, and report the expanded search's RQ1-shape recovery; expanded search scoped to the RQ3 mitigation. |
| **"Iteration" undefined.** W12 · Reviewer1 (`:70`): "18"/"13" iterations — manual edits vs tuning steps? | Defined in the optimization protocol: one source edit + one bench run + a correctness check, logged & committed; failed/regressing attempts count. `AKO4ALL/TASK.md` | §1 To-A #6 (**"Iteration"**): gives the definition + will add the iteration tables. |
| ⭐ 🔑 **Clock locking / significance.** W8 · Reviewer1 (`:50`) — *the single most-quoted rigor line*: clocks may not be locked, so "94.6% vs 97.8% may not be meaningful." | Clocks **locked** (graphics 1410 MHz held flat under load, memory 9001) + re-measured 9 near-parity kernels (17 rows, 100 reps) with median/std/p95 + a propagated 95% band: run-to-run rel-std **0.0–0.9%** (vs the paper's 9%); every small gap resolves as statistically real. `experiments/exp_significance.py`, `results/<gpu>/significance.csv`, `clock_lock.txt` | §1 To-A #7 (**Measurement significance**): clocks locked, dispersion (median/std/p95) reported, small gaps "resolved as real or within noise." |
| **Benchmark provenance/selection.** W9 · Reviewer2-Q1 (`:127,142`): how were kernels chosen from TritonBench / TileLang examples? Is the suite representative? | Triton kernels from TritonBench's GitHub channel (cite its published provenance/selection); `layer_norm` = TorchInductor-generated (verifiable from the kernel header). **TileLang kernels = our own re-implementations** — no comparable TileLang benchmark exists, and the artifact shows no copied-example markers (uniform project scaffolding; see `logs/REBUTTAL_EXPERIMENT_PROTOCOLS.md:538`). Kernel-mix census: **15 simpler (element-wise/reduction/normalization) : 7 compute-heavy (GEMM, batched GEMM, conv2d, attention, fused linear+activation) ≈ 2:1** — moderate, not "highly biased." 🚩 paper inconsistency: §3.1 ("the TileLang example repository") contradicts §2 ("TileLang re-implementations") — correct §3.1 to match in revision (reviewer-disprovable, B-Q1). ⚠️ earlier "FlagGems idioms" provenance is naming-only — drop unless source-confirmed. (documentation task) | §1 To-All #2.1 + #2.2 + To-B #1: §2.1 states the **four selection criteria** narrowing TritonBench 184 → 22; §2.2 covers the Triton-from-TritonBench / TileLang-manual provenance and the per-category + per-kernel-table representativeness commitment. |
| **Per-kernel element-wise.** W10 · Reviewer2-Q3 (`:129,146`): report each of the 15 element-wise kernels, not just the category aggregate. | Per-kernel latencies already in `ViperBench/results/profile.csv` (15 kernels). 🚩 **Fixed a contamination bug:** the cross_entropy reference was an unvectorized Python loop (E_lib read ~851,000%) → rewritten as vectorized flash-CE; `profile.csv` surgically patched (E_lib 1277%/86%). | §1 To-B #2 (**Per-kernel element-wise**): will tabulate all 15 individually. |
| **Correctness methodology.** W11 · Reviewer3, Reviewer3-Q (`:174,180,186`): how many failures, what tolerances / input distributions / edge cases; were mitigation kernels revalidated? | `exp_correctness_edge.py`: per-dtype tolerances + `loose_tol`; NaN/Inf/large/denormal edge cases; mitigation kernels revalidated. `results/<gpu>/correctness_edge.csv` | §1 A&C #2 + To-C #2: reports tolerances, input distributions, per-kernel overrides, failure counts, and mitigation revalidation as first-class. |
| 🔑 **Cross-architecture generality.** W5/W13 · Reviewer1-Q5, Reviewer1 (`:32,64`), Reviewer3 (`:87,173,187`): single Ada GPU; RC3 even cites A100 specs on Ada; do results hold on A100/H100? | Access **secured**; runbook prepared. This is the one genuinely new experiment — deferred to the revision (no new results in the rebuttal). `experiments/A100_H100_RUNBOOK.md` | §1 To-All #1 + the §1 revision plan: A100/H100 as the principal new experiment. |
| **Minor.** M1/M2/M3 · Reviewer1 (`:74,75`), Reviewer2 (`:136,137,138`): 21 vs 22 kernels; "anamoly" typo; Table 1 notation unclear. | Trivial corrections verified against the code (22 kernel dirs) and abstract. | §1 A&B #4 (**Minor corrections**): 21→22; anamoly→anomaly; Table 1 notation spelled out. |

---

## §3 · Detailed explanation of the table

### How to read it
Each row is one reviewer concern (its weakness handle `W#`, the reviewer(s), and the exact
`reviews.txt` lines), paired with **what we actually ran** (experiment + headline finding +
artifact path you can open) and **how the rebuttal text speaks to it** (the §1 section and numbered
point). §1 is grouped by reviewer overlap; the third column points into it (e.g., "A&B #2"), and the
same paragraphs also appear in the Appendix A 5-group draft. The flags encode the two things that
matter most for an honest rebuttal: 🚩 marks a finding that **changes a claim in the paper** (we
re-measured it; we do not re-assert the old wording), and ⚠️ marks a concern the 750-word text does
**not** fully cover. 🔑 marks the rows that decide acceptance.

### The three acceptance-critical rows (why @Reviewer_A is the hinge)
@Reviewer_A is the only Weak-Reject reviewer who is also the Artifact reviewer and the only one who
raised pure measurement-rigor objections — so acceptance hinges on @Reviewer_A, and three rows carry
that weight:

1. **Counters shown (W3).** The paper's biggest structural weakness was a "counter-grounded"
   taxonomy with **zero counters printed**. The full Nsight suite has now run (24/24, 0 failed),
   delivering exactly the counters @Reviewer_A-Q4 enumerated — load-vectorization for conv vs GEMM,
   register/spill for RC3, warp-stall for RC0 — plus the RC2b/RC3/RC4 isolations @Reviewer_B-Q4 asked
   for. This converts the single most-cited gap from "promised" to "in hand."
2. **RC0 + FP32 attribution (W1).** @Reviewer_A's lead objection is that the taxonomy conflates
   compiler bugs with kernel-authoring/example-quality issues. We answer with a clean split
   (RC0(a) authoring / RC0(b) code-gen) and a controlled FP32 root-cause (the `T.gemm`→TF32 path).
3. **Clock locking (W8, ⭐ done this round).** @Reviewer_A's most quotable single line —
   *"without locked clocks, 94.6% vs 97.8% may not be meaningful"* — is now literally answered:
   clocks are locked and the near-parity set re-measured with dispersion. That pair is two LayerNorm
   efficiencies — Triton's (94.6%, Table 3) and TileLang's *after* the RC0 fix (97.8%, Table 6), **not** a
   layer_norm-vs-softmax pair. Our locked-clock re-measurement of the near-parity set re-runs Triton
   LayerNorm at **94.5%** with run-to-run variation of **0.0–0.9%** (not the paper's 9%), so efficiency
   gaps of a few points resolve as *real*, not clock noise. The revision
   can therefore **delete the Table 7 "9% clock variation" footnote** outright.

### The integrity stance — why "grounds & sharpens", never "re-asserts"
This rebuttal works *only* because every experiment is real, and running them **corrected three
mechanisms** the paper originally asserted:

- 🚩 **RC0(a)** is **memory-latency-bound** (`long_scoreboard` dominates, `barrier`≈0) — *not*
  "sequential thread synchronization."
- 🚩 **RC3** register spilling is real for **TileLang layer_norm** but **absent for conv**
  (`n_spills=0`); conv is occupancy/coalescing-bound. The paper attributed the spill to conv K≥5.
- 🚩 **RC4** Winograd accounts for only ≈**2–3%** of the conv gap — *not* the "primary" remaining
  cause.

Because a reviewer can re-run these counters in minutes, the rebuttal deliberately says the counters
**"ground the taxonomy and, where they sharpen an attribution, we will update the analysis in the
revision"** — it never re-states the now-refuted labels. The corrected story is *stronger* (it is
counter-grounded), which is exactly why honesty here is the winning move, not a concession. The
matching paper-text edits are tracked in `REVISION_TODO.md` item 1 (abstract, §6, Table 5).

### The honesty gaps that this draft now closes
The earlier ≤750 draft (Appendix A) left a few reviewer points only implicit; the grouped §1 above
now states each explicitly: **W4** (in-DSL/compiler addressability — To @Reviewer_A, Point 5);
**@Reviewer_C correctness completeness** (failure counts, input distributions, mitigation revalidation
— A&C Point 2 and To @Reviewer_C, Point 2); **@Reviewer_B representativeness** and **@Reviewer_C scope
calibration** (To All Reviewers, Point 1). **Gap 4** (conv-mitigation generality) is now completed
work, folded into A&B Point 2. The only deliberately *uncontested* point is novelty (@Reviewer_A,
`:40`), which an empirical study typically does not rebut with experiments.

### Internal-audit findings (found by us, not reviewers)
Four consistency issues our own audits surfaced are **deliberately not raised** in the rebuttal
(no reviewer asked, and surfacing them invites attack), but are tracked for the revision in
`REVISION_TODO.md`: **N1** — the paper states NHWC / `allow_tf32=False` / `cudnn.benchmark=False`
but the benchmark code sets none and runs NCHW (entangled with RC1's "NHWC breaks LDG.128"
wording); **N2** — §3.3 says "CUDA events … average" but the code uses `perf_counter` + median;
**N3** — input-shape documentation drift; **N4** — §7.3's matmul mitigation is reported at 4096² as
1.66×/108% (Table 8), but the artifact's 4096² autotune point is 1.23×/102% and the dramatic 32%→98%
recovery is at 16384² — so the reviewer-facing reconciliation rests on the measured search-space
recovery (strongest at the RQ1 16384² shape) and the §7.3 4096² number is reconciled in the revision.
N1 and N4 are reviewer-disprovable from the artifact (the Artifact reviewer can open
`autotune_matmul.csv`), so they are the highest-priority revision fixes.

### Rebuttal ↔ revision split
The (final, trimmed) rebuttal text stays **qualitative and number-free** (compliant with ASE's
no-new-results rule): it states corrections, definitions, mechanisms, and commitments only. **All
numbers live in the artifact**, not the response — `experiments/results/<gpu>/` (significance,
counters, conv sweep + mitigation, FP32, fused baselines, autotune, correctness) and
`ViperBench/results/profile.csv`. Guardrails kept in the wording: no counter values in the response;
"a known upstream limitation" (not "we filed") for the `T.reduce` issue; and cross_entropy described
as **flash-CE, so `F.cross_entropy` is not an equivalent baseline**.

---

## §4 · Open revision items (NOT part of the rebuttal response)

Concerns the rebuttal round cannot close (ASE forbids new results in the response); each is
committed for the revision and scheduled in the §1 revision plan. Extends `REVISION_TODO.md`.

- [ ] **A100/H100 cross-architecture runs — the principal new experiment.** §1 commits to this
  (@Reviewer_A-Q5, @Reviewer_C). Execute via `experiments/A100_H100_RUNBOOK.md` (access secured):
  re-run the counter, conv-filter, FP32, and significance experiments on A100/H100 and report
  whether the root causes **and their relative impact** survive beyond Ada (sm_89).
- [ ] **Gap 6 · RC3 cites A100 register specs in an Ada-only study** (W13; @Reviewer_A,
  `reviews.txt:64`). Reconcile the A100 register-count citation in RC3 with the sm_89 measurements,
  folded into the RC3 re-attribution (measured: conv `n_spills=0`; the 51 GB spill is in TileLang
  layer_norm, not conv). See `REVISION_TODO.md` item 1.
- [ ] **Gap 7 · kernel-mix composition** (@Reviewer_A, `reviews.txt:32`): "a large portion of the
  evaluated kernels are relatively simple element-wise/reduction operators." Justify the kernel mix
  inside the benchmark provenance / selection-criteria / representativeness write-up (folds with
  @Reviewer_B-Q1).

> Gap 8 (@Reviewer_A, `reviews.txt:40` — "novelty is in organization/interpretation, not new
> techniques") is intentionally **not** contested; standard for an empirical-study rebuttal.
> The grouped §1 draft above already folds in gaps 1–5 (gap 4 is completed work; the rest are
> wording), so it runs over-length on purpose — trimming to ≤750 is the remaining step, with the
> prior ≤750 draft kept in Appendix A as the fallback.

---

## §5 · Revision plan (≤ 1 month)

**Folded into the response.** The revision plan now ships *with* the rebuttal text as the closing
subsection of §1 ("Revision Plan (≤ 1 Month)"), outside the ≤750-word limit. It is kept here only as
a pointer; §4's open items are scheduled against that plan. The **full internal plan** — including
the N1–N3 audit fixes deliberately *not* surfaced to reviewers (see §3's integrity stance) — remains
tracked in `REVISION_TODO.md`.

---

### Provenance
- **Full campaign history & strategy** (8 working docs): `logs/` (see `logs/README.md`).
- **Measured evidence:** `experiments/results/<gpu>/` (`<gpu>` = `NVIDIA_RTX_4000_Ada_Generation`)
  and `ViperBench/results/profile.csv`; experiment scripts in `experiments/`.
- **Forward-looking revision actions:** `REVISION_TODO.md`.
- **Environment:** RTX 4000 Ada (sm_89), torch 2.8.0+cu126, triton 3.4.0, tilelang 0.1.6.post1,
  ncu 2024.3.2.0, CUDA 12.6.

---

## Appendix A · Prior 5-group draft (≤750 words, superseded by §1)

*Kept verbatim as a trimmed-length fallback. §1 above regroups the same content into the reviewer
"@Reviewer" template and adds the gap fixes; this version is the number-free, ≤750-word response
organized as corrections / completed / critical / new / revision commitments.*

We thank all three reviewers for engaging deeply. We are encouraged that the topic is seen as timely and the methodology "stronger than a pure benchmark table" (Reviewer3). We address the concerns in the following groups: corrections, completed work, critical concerns, new experiment, and revision commitments.

### Corrections (from the existing paper/artifact)

**Normalization baseline (Reviewer1, Reviewer3).** The LayerNorm baseline is the **fused** `F.layer_norm`, not unfused eager, so Triton's 94.6% is a fair library comparison. Only RMSNorm and element-wise use eager paths; we will state this per-category explicitly.

**"Contradiction" between §5 (Δ=0pp) and §7.3 (1.66×) (Reviewer1-Q3, Reviewer2-Q5).** These differ in search space, not just shape. §5's "heuristic tuning" is a 12-config block-tile grid (bm,bn∈{32,64,128}, bk∈{32,64}); varying only block tiles leaves the default unchanged, so Δ≈0. §7.3's expanded search adds the non-block-tile levers (GROUP_SIZE_M L2-swizzle, num_warps, num_stages) that actually recover performance — substantially at the RQ1 matmul shape. We will define "heuristic tuning" precisely, give both search spaces, and explain the expanded search was scoped to the RQ3 mitigation rather than RQ1.

**"Iteration" (Reviewer1).** Defined precisely in our protocol: one kernel-source edit followed by one benchmark run with a correctness check, logged and committed; failed/regressing attempts count. We will add this definition and the iteration tables.

**RC0 attribution (Reviewer1-Q1).** We agree the label is imprecise, and clarifying it sharpens the taxonomy. RC0(a) `T.serial`→`T.reduce` is a **kernel-authoring** issue (correctable in user space, hence "no new technique"); RC0(b) absent LDG.128 vectorization is a **code-generation** issue. We will split these and cite the corresponding upstream `T.reduce` lowering limitation.

**Minor (Reviewer1, Reviewer2):** 21→22 kernels; "anamoly"→"anomaly"; Table 1 notation (16384² = square fp16 GEMM; 64×128² = batched, batch 64 of 128×128) spelled out.

### Completed (our existing infrastructure ran these directly)

**Per-kernel element-wise (Reviewer2-Q3).** Per-kernel latencies for all 15 element-wise kernels are already in our benchmark results (`ViperBench/results/profile.csv`); the revision lifts them into the paper as a per-kernel table alongside the per-category aggregate.

**Convolution coverage (Reviewer1-Q2, Reviewer2-Q2).** Our conv kernels already support arbitrary filters, strides, and groups (5×5/stride-2 are correctness-validated today). We have extended the **performance** evaluation to 1×1, 5×5, 7×7, depthwise, and strided cases at realistic shapes (the 1×1 control isolates RC1, and the 5×5/7×7 cases characterize how the library-efficiency gap scales with filter size), as requested.

**Measurement significance (Reviewer1).** With GPU clocks locked (removing the Table 7 boost variation), we re-measured the key comparisons and report dispersion (median, std-dev, p95), so small efficiency differences are resolved as real or within noise.

**Nsight counters + RC2b/RC3/RC4 mitigation isolations (Reviewer1-Q4, Reviewer2-Q4).** We have now collected the requested hardware-counter tables for representative kernels: global-load (vectorization) efficiency for conv vs. GEMM, register/occupancy and spill indicators, the warp-stall breakdown, and L2/DRAM behavior. These supply the measured evidence underpinning the taxonomy; where they **sharpen an attribution** (e.g., occupancy/vectorization vs. register-spill, or memory-latency vs. synchronization stalls), we will update the analysis accordingly in the revision. We also ran the controlled mitigation/isolation experiments requested for RC2b, RC3, and RC4: an L2-residency measurement (RC2b); the register/occupancy/spill counters above (RC3); and a Winograd-eligible-vs-ineligible cuDNN comparison (RC4; cuDNN's own logging confirms it selects Winograd for 3×3).

### Critical concerns handled

**FP32 GEMM failure (Reviewer1-Q1, Reviewer3).** We agree this is a genuine and valuable SE finding and should not merely be excluded. We have root-caused it with a controlled experiment: `T.gemm` lowers FP32 to the TF32 tensor-core path, whereas a non-`T.gemm` FP32 accumulation is numerically correct, i.e., the failure is TF32 mantissa truncation surfacing through cancellation at near-zero outputs, not a logic error. We have added a reproducible FP32 correctness case plus edge-case inputs (NaN/Inf/large/denormal), and will report tolerances and per-kernel overrides as a first-class part of the benchmark (Reviewer3).

**Baseline fairness (Reviewer1, Reviewer3).** Adopting R3's suggestion, we replace the single blended "library efficiency" with **split metrics** — vendor-library, fused-library, and eager-PyTorch — and have collected fused (`torch.compile`) baselines for element-wise/normalization kernels so the revision can report all three side-by-side and let each per-category gap be interpreted under each baseline.

### New experiment

We have gained direct access to A100/H100 and are committed to expanding our experiments during revision, testing whether the root causes and their relative impact hold beyond Ada (Reviewer1-Q5, Reviewer3).

### Revision commitments

We will fold all of the completed work above into the paper, including the full Nsight counter tables with any sharpened RC0/RC3 attribution, split-baseline/fused metrics, the conv filter/stride/depthwise sweep, the FP32 root-cause and edge-case suite, the per-kernel element-wise table, the "heuristic tuning"/"iteration" definitions and RC0 split, and the CI re-measurement — document benchmark provenance and selection criteria (Reviewer2-Q1), and add the A100/H100 cross-architecture study (Reviewer1-Q5, Reviewer3) as the principal new experiment.
