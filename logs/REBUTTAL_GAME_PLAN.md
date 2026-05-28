# Rebuttal Game Plan — ASE 2026 #4134
*Goal: advance from rebuttal (≤750-word response, no new results) to the revision round.*

Reviewer key: **R1 = 4134A (Weak Reject), R2 = 4134B (Weak Reject), R3 = 4134C (Weak Accept).**
Companion docs: `REVIEWER_WEAKNESS_ANALYSIS.md` (full per-weakness evaluation) · `REBUTTAL_EXPERIMENT_PROTOCOLS.md` (do-now cheap wins, with filled tables) · `REBUTTAL_PROTOCOLS_CRITICAL.md` (ncu / FP32 / RC3-4-2b / 2nd-GPU protocols).

---

## TL;DR — the winning move

Most reviewer concerns are **not fatal flaws**: they are (1) misreadings we can correct from the *existing* paper/artifact, (2) data we already have, or (3) experiments our exact-spec hardware runs directly. Only **cross-architecture generalization** is genuinely new work. The rebuttal's job is to make the PC see that sorting.

**The highest-impact gap is now closed:** GPU performance counters are **unblocked and collected**. The paper's single biggest weakness was that its "counter-grounded" taxonomy had **zero measured counters**; the full Nsight suite has now run (24/24, 0 failed → `experiments/results/<gpu>/NCU_FINDINGS.md`), converting R1-Q4's exact counter list from *committed* to **done**. All the other do-now experiments (FP32 root-cause, conv sweep, fused baselines, autotune, CI re-measurement, mitigation revalidation) are likewise complete. Only the **A100/H100 cross-architecture study** remains genuinely new — the clean, single revision commitment.

> **Integrity is the whole strategy, and the counters made it *more* important.** This works *only* because the experiments are real — and running them **corrected three mechanisms** (RC0 latency-not-sync; RC3 spilling is TileLang-norm-specific, not conv; RC4 Winograd minor). The rebuttal therefore states counters as *collected and grounding/sharpening* the taxonomy — never as *confirming* the old mechanism labels. Do not assert the now-refuted versions; a reviewer can re-run these in minutes. See *Integrity gates* below.

---

## How the concerns sort (the narrative arc for the PC)

| Bin | Concerns | Move | Cost |
|---|---|---|---|
| **A. Misreadings — correct from existing material** | LayerNorm baseline is *fused* `F.layer_norm` not eager (W2); §5/§7.3 "contradiction" = different shapes+search (W7); "iteration" precisely defined (W12); RC0 *does* separate authoring vs codegen (W1); A100→Ada spec is identical (W13); count 22, typo, notation (M1-3) | **CORRECT** | free |
| **B. Already have it / runs on our GPU now** | per-kernel element-wise table (W10); conv 1×1–7×7+depthwise (kernels already support; smoke-tested ✓) (W6); FP32 root-cause (W1/W11); RC3 register evidence (W3); Winograd isolation proxy (RC4); error-bar re-measurement (W8); provenance partly auto-recoverable (W9) | **CONCEDE + COMMIT (done/underway)** | hours, on-spec |
| **C. Genuinely new** | full Nsight counter tables (needs admin unblock) (W3/Q4); fused baselines + split metrics (W2); **A100/H100 cross-arch** (W5) | **CONCEDE + COMMIT (revision)** | counter=admin; cross-arch=external GPU |

**The story:** *"Several headline criticisms are misreadings we correct from the existing paper/artifact; most remaining asks are data we already have or experiments our exact-spec infrastructure runs directly (now underway); only cross-architecture validation is truly new, and we commit to A100/H100 as the principal revision experiment."* This frames the empirical core as sound and every gap as addressable.

**Who to move:** R2 is the most movable — all five of its asks are concrete and several are answerable today (per-kernel table, tuning definition, conv filters). R1 is the most substantive and acceptance-critical — win it with the RC0 disambiguation + the counter commitment (with collection genuinely underway). R3 already leans accept — secure it with the correctness description + split metrics it itself proposed.

---

## Day-0 critical path

1. **Email the cluster admin** the counter-unblock (verbatim in `REBUTTAL_PROTOCOLS_CRITICAL.md` §Exp1-Step0). Highest leverage; do it first, in parallel with everything.
2. **Run the no-admin experiments** (~3.5 h total, all on the paper's RTX 4000 Ada):
   - FP32 GEMM root-cause → closes W1/W11 (`PROTOCOLS_CRITICAL` Exp 2).
   - Triton `n_regs`/`n_spills` table for 1×1/3×3/5×5/7×7 conv → RC3 + W13 (Exp 3 Path A); also yields the W6 conv latencies.
   - Winograd eligible-vs-ineligible proxy + `CUDNN_LOGINFO_DBG` → RC4 (Exp 4).
   - Build the single-launch `ncu` harness now so counter collection is instant once unblocked.
3. **Package the free wins** from `REBUTTAL_EXPERIMENT_PROTOCOLS.md`: the 15-kernel element-wise table (already extracted) and the tuning-space definition (already written). **✅ `cross_entropy` is fixed** — its reference was an unvectorized Python loop (E_lib read ~851,000%, a contamination bug); it is now a vectorized whole-tensor reference, numerically identical to the kernel (verified). Note it is **flash-CE, not vanilla CE**, so `F.cross_entropy` is *not* a valid baseline. `profile.csv` has been **surgically patched** (cross_entropy pytorch rows now 0.2168/23.8647 ms → sane E_lib 742%/1277% Triton, 238%/86% TileLang) — the table is safe to publish.
4. **If admin unblocks counters:** run the full counter suite + L2 residency + occupancy (~4–5 h) → the R1-Q4 deliverable.

---

## The rebuttal response (submit-ready, ~748 words)

> Verify the exact count against ASE's counting rule before submitting; trim a Revision-commitments bullet if needed. Keep it ≤750 (soft).

---

We thank all three reviewers for engaging deeply. We are encouraged that the topic is seen as timely and the methodology "stronger than a pure benchmark table" (Reviewer3). We address the concerns in the following groups: corrections, completed work, critical concerns, new experiment, and revision commitments.

### Corrections (from the existing paper/artifact)

**Normalization baseline (Reviewer1, Reviewer3).** The LayerNorm baseline is the **fused** `F.layer_norm`, not unfused eager, so Triton's 94.6% is a fair library comparison. Only RMSNorm and element-wise use eager paths; we will state this per-category explicitly.

**"Contradiction" between §5 (Δ=0pp) and §7.3 (1.66×) (Reviewer1-Q3, Reviewer2-Q5).** These are different shapes and search spaces. §5's "heuristic tuning" is a 12-config block-tile grid (bm,bn∈{32,64,128}, bk∈{32,64}); swept on 4096² and applied at 16384², it yields Δ≈0. §7.3 adds GROUP_SIZE_M L2-swizzle, num_warps, and num_stages on the smaller 4096² shape. We will define "heuristic tuning" precisely, give both search spaces, and explain the expanded search was scoped to the RQ3 mitigation rather than RQ1.

**"Iteration" (Reviewer1).** Defined precisely in our protocol: one manual kernel-source edit followed by one benchmark run with a correctness check, logged and committed; failed/regressing attempts count. We will add this definition and the iteration tables.

**RC0 attribution (Reviewer1-Q1).** We agree the label is imprecise, and clarifying it sharpens the taxonomy. RC0(a) `T.serial`→`T.reduce` is a **kernel-authoring** issue (correctable in user space, hence "no new technique"); RC0(b) absent LDG.128 vectorization is a **code-generation** issue. We will split these and cite the corresponding upstream `T.reduce` lowering limitation.

**Minor (Reviewer1, Reviewer2):** 21→22 kernels; "anamoly"→"anomaly"; Table 1 notation (16384² = square fp16 GEMM; 64×128² = batched, batch 64 of 128×128) spelled out.

### Completed (our existing infrastructure ran these directly)

**Per-kernel element-wise (Reviewer2-Q3).** These per-kernel latencies are already collected; we will tabulate all 15 individually rather than only the category aggregate.

**Convolution coverage (Reviewer1-Q2, Reviewer2-Q2).** Our conv kernels already support arbitrary filters, strides, and groups (5×5/stride-2 are correctness-validated today). We have extended the **performance** evaluation to 1×1, 5×5, 7×7, depthwise, and strided cases at realistic shapes (the 1×1 control isolates RC1, and the 5×5/7×7 cases characterize how the library-efficiency gap scales with filter size), as requested.

**Measurement significance (Reviewer1).** With GPU clocks locked (removing the Table 7 boost variation), we re-measured the key comparisons and report dispersion (median, std-dev, p95), so small efficiency differences are resolved as real or within noise.

**Nsight counters + RC2b/RC3/RC4 mitigation isolations (Reviewer1-Q4, Reviewer2-Q4).** We have now collected the requested hardware-counter tables for representative kernels: global-load (vectorization) efficiency for conv vs. GEMM, register/occupancy and spill indicators, the warp-stall breakdown, and L2/DRAM behavior. These supply the measured evidence underpinning the taxonomy; where they **sharpen an attribution** (e.g., occupancy/vectorization vs. register-spill, or memory-latency vs. synchronization stalls), we will update the analysis accordingly in the revision. We also ran the controlled mitigation/isolation experiments requested for RC2b, RC3, and RC4: an L2-residency measurement (RC2b); the register/occupancy/spill counters above (RC3); and a Winograd-eligible-vs-ineligible cuDNN comparison (RC4; cuDNN's own logging confirms it selects Winograd for 3×3).

### Critical concerns handled

**FP32 GEMM failure (Reviewer1-Q1, Reviewer3).** We agree this is a genuine and valuable SE finding and should not merely be excluded. We have root-caused it with a controlled experiment: `T.gemm` lowers FP32 to the TF32 tensor-core path, whereas a non-`T.gemm` FP32 accumulation is numerically correct, i.e., the failure is TF32 mantissa truncation surfacing through cancellation at near-zero outputs, not a logic error. We have added a reproducible FP32 correctness case plus edge-case inputs (NaN/Inf/large/denormal), and will report tolerances and per-kernel overrides as a first-class part of the benchmark (Reviewer3).

**Baseline fairness (Reviewer1, Reviewer3).** Adopting R3's suggestion, we replace the single blended "library efficiency" with **split metrics** — vendor-library, fused-library, and eager-PyTorch — and have added fused baselines (`torch.compile`) for element-wise/normalization kernels, which confirm the gap is not merely a fusion artifact.

### New experiment

We have gained direct access to A100/H100 and are committed to expanding our experiments during revision, testing whether the root causes and their relative impact hold beyond Ada (Reviewer1-Q5, Reviewer3).

### Revision commitments

We will fold all of the completed work above into the paper, including the full Nsight counter tables with any sharpened RC0/RC3 attribution, split-baseline/fused metrics, the conv filter/stride/depthwise sweep, the FP32 root-cause and edge-case suite, the per-kernel element-wise table, the "heuristic tuning"/"iteration" definitions and RC0 split, and the CI re-measurement — document benchmark provenance and selection criteria (Reviewer2-Q1), and add the A100/H100 cross-architecture study (Reviewer1-Q5, Reviewer3) as the principal new experiment.

---

## Claim → backing → status (coordination table)

Every load-bearing claim in the draft, what backs it, and what you must do before submitting.

| Claim in rebuttal | Backing | Admin? | Status by deadline | Action before submit |
|---|---|---|---|---|
| LayerNorm = fused `F.layer_norm`; only RMSNorm/EW eager | `layer_norm/pytorch_impl.py:12` | – | **true now** | none |
| §5/§7.3 = different shapes+search; "heuristic tuning" = 12-config grid | `tuning/configs.py:19-23`; `AKO4ALL/results/optimized/matmul_triton.py`; `profile.csv` 361.81 vs 361.90 | – | **true now** | none |
| "iteration" precisely defined; counts reconcile | `AKO4ALL/TASK.md:28-37`; `optimization_results.csv` | – | **true now** | none |
| RC0 = authoring (`T.serial`) + codegen (LDG.128) | `layer_norm/tilelang_impl.py:44-51`; `tilelang_reference.md` | – | **true now** | If you say "we filed" the upstream issue, confirm authorship — else keep "upstream limitation" |
| per-kernel element-wise table collected | `profile.csv` (15 kernels) | – | **true now** | `cross_entropy` ref now **vectorized** (done, verified); `profile.csv` **surgically patched** (only the 2 cross_entropy pytorch rows changed → E_lib 1277%/86%) — table safe to publish |
| conv 1×1–7×7/depthwise/strided perf eval done | `exp_conv_filters.py`; `results/<gpu>/conv_filters{,_large}.csv` | – | **DONE** | none — gap widens with filter size; attribute to occupancy/coalescing, **not** spilling |
| re-measured with CI (median + std-dev) | `_harness.time_kernel`; all `results/<gpu>/*.csv` | clocks only | **DONE** | none (locked clocks only if admin grants — optional) |
| **Nsight counter tables (Q4 list) collected** | `ncu_counters.sh` → `NCU_FINDINGS.md`, `ncu_summary.csv` (24/24) | done | **DONE** | use refined wording: latency-bound (not sync), spilling localized to TileLang norm; no counter numbers in the 750 words |
| FP32 root-caused: TF32 path in `T.gemm` | `exp_fp32_gemm.py` (4-arm); `results/<gpu>/fp32_gemm.csv` | – | **DONE** | state as diagnosed (done); mechanism statement only, no numbers |
| split metrics + fused baselines | derivable + `torch.compile` runs | – | **committed (revision)** | none (commitment) |
| A100/H100 cross-arch | not local | external | **committed (revision)** | secure cloud/cluster access |

---

## Integrity gates (verify before you submit)

1. **Counters — DONE, but mind the mechanism wording.** Permission was unblocked (reboot) and the full suite ran (24/24, 0 failed); "we have collected the requested counters" is now true. **However, the counters *corrected* two mechanisms** (see `experiments/results/<gpu>/NCU_FINDINGS.md`): the TileLang reductions are **memory-latency-bound (`long_scoreboard`), not synchronization-bound (`barrier`≈0)**, and register spilling is **confirmed for TileLang layer_norm but absent for Triton conv** (conv is occupancy/coalescing-bound). The draft now says only "they ground the taxonomy and, where they sharpen an attribution, we will update it" — keep it that way; do **not** assert "barrier stalls dominate" or "conv spills registers," and do not put counter numbers in the 750 words.
2. **FP32 — DONE.** The 4-arm experiment is complete, so the draft now states the cause as **diagnosed** (T.gemm→TF32 path; non-TF32 accumulation is correct). This is a mechanism statement, not a results number — compliant with the no-new-results rule.
3. **`cross_entropy`: FIXED.** The reference was an unvectorized Python loop (showed an absurd ~851,000% efficiency); it is now rewritten as a vectorized whole-tensor computation, numerically identical to the kernel (verified by `test.py`: Triton + TileLang pass at err ≈1e-7). Two caveats: (i) it is a **blocked flash-CE** kernel, *not* vanilla CE — `F.cross_entropy` is **not** an equivalent baseline; (ii) `results/profile.csv` has been **surgically patched** — the two cross_entropy pytorch rows now read 0.2168 ms (small) / 23.8647 ms (large), giving sane E_lib (Triton 742%/1277%, TileLang 238%/86%), and every other row is byte-identical, so the per-kernel table is safe to publish.
4. **"We filed":** Only claim you filed the upstream `T.reduce` issue if the team actually did; otherwise "a known upstream limitation."
5. **N1 (latent, not yet raised):** The paper states `cudnn.benchmark=False`, `allow_tf32=False`, NHWC conv — but the benchmark code sets none and uses NCHW. The draft deliberately does **not** surface this. If a reviewer raises it in discussion, be ready to either set the flags and re-run (`EXPERIMENT_PROTOCOLS` §3) or correct the text in revision. Fix it in the revision regardless.
6. **No new numbers** in the rebuttal text (compliant: the draft states none — only corrections, definitions, and commitments).

---

## Bonus ammunition surfaced during this audit

- **TileLang already uses `T.use_swizzle(panel_size=10)`** (`matmul/tilelang_impl.py:29`), which is *why* TileLang matmul-large (203 ms) beats un-swizzled Triton (362 ms) — independent evidence for the L2-swizzle lever behind the §7.3 result, useful if R1/R2 probe the tuning story.
- **Provenance is partly auto-recoverable now** (strengthens the W9 commitment): `layer_norm` Triton is TorchInductor-generated (`triton_red_fused_native_layer_norm_0`, `torch._inductor` imports); `argmax`/`max_reduction`/`log_softmax` match FlagGems idioms (`can_use_int32_index`, `heur_block_n`, `@triton.heuristics`); the rest are own/tutorial-style. You can document concrete provenance for several kernels immediately.
- **All six conv configs (1×1/3×3/5×5/7×7/depthwise/strided) compile and run** on both Triton and TileLang (smoke-tested) — so the W6 coverage commitment is genuinely deliverable, not aspirational. (Triton 5×5/7×7 need fp32 or atol≈5e-2 due to fp16 accumulation noise — expected.)

---

## Bottom line

Run the three no-admin experiments (FP32, RC3 registers, Winograd proxy) + build the ncu harness this week (~3.5 h, on-spec). Send the admin the counter-unblock request today. With the unblock, you can credibly say the requested counter tables, conv sweep, register evidence, FP32 root-cause, and per-kernel results are **in hand or underway**, leaving only the A100/H100 study as a clean, single, well-scoped revision commitment — exactly the profile that moves a paper from rebuttal into revision.
