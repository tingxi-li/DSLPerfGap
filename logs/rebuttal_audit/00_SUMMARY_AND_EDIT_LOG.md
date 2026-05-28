# Rebuttal Audit — Summary & Edit Log (ASE 2026 #4134)

Coordinated multi-agent audit of `REBUTTAL.md` against the artifacts, the reviews, and the paper,
then edited `REBUTTAL.md`. This file is the capstone; per-agent reports are alongside
(`A_artifact_factcheck.md`, `B_reviewer_concerns.md`, `C_inline_and_census.md`, `D_meta_audit.md`).

## Roles (who did what)
- **A · Artifact Fact-Checker** — every checkable claim in `REBUTTAL.md` vs `experiments/results/.../*.csv`, `NCU_FINDINGS.md`, `profile.csv`, kernel code. (your step 1)
- **B · Reviewer Comment Analyst** — every reviewer concern from `reviews.txt` + the paper PDF; reviews.txt citation accuracy; coverage. (your step 2)
- **C · Inline-Comment Auditor + Census** — the embedded author `>` TODOs; the 22-kernel simple/complex census. (your step 4)
- **D · Independent Meta-Auditor** — stress-tested A/B/C and the planned edits against primary sources. (your step 5)
- **Coordinator (me)** — cross-match (your step 3), first-hand re-verification of the two consequential items (PDF Tables 3/6, `autotune_matmul.csv`, `exp_autotune_matmul.py`), and the edits.

## Headline verdict
The rebuttal's load-bearing 🚩 "corrects-paper" claims (RC0a memory-latency-bound; RC3 spill is
TileLang layer_norm not conv; RC4 Winograd ≈2–3%) are **artifact-solid** — derived `NCU_FINDINGS.md`
matches raw `ncu_summary.csv`, 0 contradictions. Every reviewer concern is addressed somewhere, and
all `reviews.txt:<line>` citations are accurate (0 off-by-N). **Two genuine factual errors** and a set
of inline-comment fixes were found and corrected (below).

## Cross-match (step 3): reviewer concern → addressed? → artifact-supported?
| Concern (reviewer) | Addressed | Artifact support | Action |
|---|---|---|---|
| RC0/FP32 attribution (A1) | yes | SUPPORTED (NCU, fp32_gemm.csv) | none |
| Baseline asymmetry (A2/C3) | yes | SUPPORTED (fused_baselines.csv) | none |
| Counters + RC2b/RC3/RC4 (A3/B3) | yes | SUPPORTED (24/24) | none |
| Kernel-mix bias (A5) | partial→**now stronger** | census 15:7 | **edited** (E3) |
| Conv coverage (A8/B2) | yes | SUPPORTED (conv_*.csv) | depthwise wording **edited** (E6) |
| Tuning §5↔§7.3 (A9/B4) | yes but **mis-stated** | shape error + paper↔artifact gap | **edited** (E2) |
| Clock locking (A10) | yes; **§3 misidentified the numbers** | SUPPORTED (significance.csv) | **edited** (E1) |
| Correctness (A11/C2) | yes | SUPPORTED (correctness_edge.csv) | none |
| Provenance/representativeness (B1/A5) | yes | layer_norm=TorchInductor solid; FlagGems shaky | **edited** (E3/E4) |
| Cross-arch (A-Q5/C) | deferred (honest) | RUNBOOK only | none |
| Minor 21→22 / anomaly / Table 1 (A,B) | yes | SUPPORTED (22 dirs) | none |

## Verified truth on the two contested items (primary-source confirmed by me)
1. **"94.6% vs 97.8%"** (reviews.txt:50, illustrative pair). PDF: **94.6% = Triton LayerNorm (Table 3, p6)**;
   **97.8% = TileLang LayerNorm *after* the RC0 fix (Table 6, p8)**. Not "layer_norm vs softmax" (Agent A's
   guess) and not "same kernel pre/post" (Agent B's guess) — pre-fix TileLang LayerNorm is 0.32%, and
   `significance.csv` contains **no 97.8%** (it re-runs only the unmitigated TileLang LayerNorm at 0.32%).
2. **Tuning** (`autotune_matmul.csv`): the **32%→98%** recovery is at **16384²** (32.23→98.49), while 4096²
   is 82.98→102.21 (**1.23×**). The paper's §7.3/Table 8 report **1.66×/108% at 4096²** — the artifact does
   not reproduce that 4096² point. The experiment's own docstring hypothesized "16384² heuristic already
   optimal" but the data refutes it (expanded search gives 3.06× there). True reconciliation = **search-space, not shape**.

## Edits applied to REBUTTAL.md
| ID | Where | Change |
|---|---|---|
| E1 | §3 clock-locking item 3 | Corrected the 94.6/97.8 identities (Triton LayerNorm / post-fix TileLang LayerNorm); scoped the locked re-measurement to Triton LayerNorm 94.5%. |
| E2 | §1 To-A&B #3, §2 tuning row, §3 N4, Appendix A | Reframed §5↔§7.3 as **search-space not shape**; fixed the 32→98% shape (16384², not 4096²); flagged the paper↔artifact 1.66× gap as new internal-audit item **N4**. |
| E3 | §1 To-All #2 + §2 provenance row | Representativeness: "not skewed to easy kernels," census **15 simpler : 7 compute-bound ≈ 2:1**, per-kernel-table commitment, accurate competitiveness (kept "DSLs competitive on simpler ops" for Triton; noted TileLang is uneven even there). **Did NOT** adopt the margin-note claim "DSLs better on complex kernels" — the data contradicts it (Triton simple-median 99% > complex 55%). |
| E4 | §1 To-All #2 + §2 provenance row | Provenance now leans on **TritonBench's published provenance**; kept layer_norm=TorchInductor (proven); dropped the shaky "FlagGems idioms" claim (naming-only). |
| E5 | §1 Claim Scope | "roughly two configurations per cell" → "two configurations per cell" (reviews.txt:173). |
| E6 | §1 To-A&B #2 | Depthwise exclusion made forward-looking + concrete ("its lowering is `groups==1`"; revision states it in Table 2, correcting §3.1). |
| E8 | §2 W3 row | Added Reviewer A's sharpest RC3/RC4 lines (`reviews.txt:58,60`) to the citation. |
| E9 | §1 | Removed the 4 resolved author `>` TODOs (L34-37, 43, 59, 65) once their fixes landed. |

All §1 response edits stay **qualitative/number-free** (numbers live in §2/§3 per the doc's own
ASE-compliance guardrail). Edits verified: inline TODOs gone, old errors gone, fixes present, table
structure intact.

## Left for the author (cannot auto-resolve)
- **`[project-website URL]`** (§1 revision plan): a placeholder — fill the real URL or drop the clause.
- **Revision items** now tracked in the rebuttal: **N4** (reconcile §7.3's 4096² 1.66×/108% with the
  artifact's 1.23×/102%) — reviewer-disprovable from `autotune_matmul.csv`, so high priority alongside N1.
- Optional: if you keep the "FlagGems" provenance, confirm it against the kernel sources first (currently naming-only).

---

## Round 2 — author inline notes on §1 "To All Reviewers" (3 issues)

| Issue | Finding | Edit |
|---|---|---|
| **#1 Soundness tone** ("are beginning the cross-architecture study" → reads as "not ready / start over") | Rhetorical only. | Reframed To-All #1: lead with "findings validated on Ada; root causes are architectural mechanisms, not Ada quirks," then position A100/H100 as a resourced generality check (access secured + runbook prepared), reported in revision. No "beginning/starting" language. |
| **#2a TileLang provenance** ("we didn't reuse any TileLang impl; no TileLang benchmark exists" + "check manual vs agentic") | **Verified: author is right.** No `tilelang_impl.py` carries any copied-example/attribution marker; all use this project's own scaffolding + self-authored design-rationale docstrings. Paper **§2 itself says "TileLang re-implementations"**; only **§3.1 says "TileLang example repository"** — a paper-internal §2↔§3.1 contradiction. **Manual vs agentic CANNOT be determined** from code/logs (uniform boilerplate, no generation signature — `logs/REBUTTAL_EXPERIMENT_PROTOCOLS.md:538` says origin "cannot be inferred from code; author must record"). | To-All #2 prose now: TileLang kernels = "our own re-implementations… no comparable TileLang benchmark exists." §2 row records the finding + flags §3.1 for correction (reviewer-disprovable, B-Q1). **Authoring method deliberately NOT stated in the rebuttal** (no evidence + strategically sensitive — author's call). |
| **#2b "compute-bound class" / "headline"** | "compute-bound class" is not one of the paper's 5 categories; "headline" was undefined. | Rewrote To-All #2 representativeness using the **paper's five real categories** (GEMM, convolution, attention, normalization, element-wise/reduction) and replaced "headline" with the concrete **aggregate "Overall" efficiency** number. |
| **#3 Claim Scope vacuous** ("which concern? no useful info") | Responds to @Reviewer_C's "calibrate claims to the design space" (reviews.txt:178); paper already has Limitations §8.2. | Rewrote To-All #3 to name @Reviewer_C's concern, cite §8.2, and state the concrete action (qualify each general claim to the measured scope). |

**Open for author (Round 2):**
- **TileLang authoring method (manual vs agentic):** only you know — the artifact can't tell. Decide whether/how to disclose it (kept out of the rebuttal for now).
- **§3.1 wording:** the revision must change "the TileLang example repository" → "TileLang re-implementations" to match §2 and the rebuttal.
- The revision-plan subsection (later in §1) still phrases cross-arch as "runs getting under way"; align its tone with the reframed To-All #1 if desired.
