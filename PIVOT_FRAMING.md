# PIVOT_FRAMING.md — Canonical framing for the pivoted paper

> **Single source of truth.** As of 2026-06-25 the paper is being completely re-framed.
> Where any other doc (paper `.tex`, `README.md`, `CLAUDE.md`, memory, `AKO4ALL/`, `experiments/`)
> conflicts with this file, **this file wins**. Reconcile the other doc, don't fork the framing.

## The pivot in one sentence

From **"An Empirical Study of GPU Kernel Performance Gaps in Modern DSLs"** (characterize → root-cause →
mitigate a performance gap) to **the DSL kernel *evaluation* problem**: developers and LLM kernel
generators have no reliable way to tell whether a DSL kernel is *well-written*, the benchmarks they
rely on pass correct-but-slow kernels, and a comprehensive benchmark is infeasible — so we characterize
the hidden gap and distill practical evaluation heuristics + optimization patterns to guide development.

**The old gap study is now *evidence inside* the new story (it is RQ2), not the thesis.**

## Why we pivoted (rationale on file)

1. The old thesis had two fatal liabilities: (a) a 300× slow kernel is hard to defend as a *realistic*
   example, and (b) reviewers said the novelty was "mainly organization/interpretation" (Reviewer A).
2. The new framing **converts both liabilities into assets**: the 300× kernel is no longer a claim about
   typical performance — it is *proof that a correct kernel passes existing benchmarks yet is 300× slow*
   (a benchmark-validity result); and "benchmark inadequacy + evaluation heuristics" is a genuine new
   SE/evaluation-methodology contribution, which is the novelty reviewers said was missing.
3. It is timely: KernelBench-style benchmarks gate LLM kernel generation; if you cannot evaluate kernel
   quality you cannot guide generation. Reviewer C explicitly flagged the LLM-kernel-gen relevance.

## New thesis (canonical paragraph)

DSLs such as Triton and TileLang are promoted as delivering near-vendor-library performance with far
lower effort, and are increasingly the lowering target for compilers (`torch.compile`) and LLM kernel
generators. But a developer who writes a *functionally correct* DSL kernel has no dependable signal for
whether it is *performant*: the kernel can pass existing benchmarks and still be 5–300× slower than the
vendor library. Prevailing DSL/LLM-kernel benchmarks (KernelBench, TritonBench) gate on correctness (and
at most a weak speedup-over-eager check) and therefore admit performance-poor kernels, while under-
covering DSLs, dtypes, shapes, and hardware and only partially preventing reward hacking. A benchmark
comprehensive enough to close this gap is combinatorially infeasible. As a pragmatic alternative we (a)
characterize where and why the hidden gap arises — separating user-space *authoring* causes from
*code-generation* and *library-maturity* causes — and (b) distill it into lightweight, partly
baseline-independent evaluation heuristics and a small set of recurring optimization patterns that guide
DSL kernel development absent a ground-truth benchmark.

## New research questions (canonical wording — adjust here, then propagate)

- **RQ1 — The evaluation gap.** Do existing DSL and LLM-generated-kernel benchmarks distinguish
  well-written kernels from performance-poor ones, and along which dimensions (DSL coverage, dtype,
  input shape, hardware, reward-hacking prevention) do they fall short?
- **RQ2 — The hidden gap and its causes.** For kernels that pass existing correctness-style benchmarks,
  how large and structured is the performance gap to vendor libraries, and which authoring,
  code-generation, and library-maturity factors cause it?
- **RQ3 — Guidance without a comprehensive benchmark.** Absent a comprehensive benchmark, can lightweight
  evaluation heuristics (comparability with the vendor baseline + a roofline anchor) and recurring
  optimization patterns reliably flag poor kernels and guide fixes? Is there a single dominant fix, and
  where are DSLs genuinely limited?

## Candidate titles (decision 2026-06-25: KEEP current `main.tex` title for now; revisit before submission)

The title change is intentionally deferred so the body reframe lands first. These stay on the table:

- *How Do You Know Your GPU Kernel Is Fast? The Evaluation Gap in DSL Kernel Development*
- *Passing Is Not Fast: Why DSL Kernel Benchmarks Miss Performance, and How to Guide Development Without One*
- *Knowing When a DSL Kernel Is Good Enough: The Evaluation Gap in GPU Kernel Generation*

## Evidence map — what transfers, what is new

| New RQ | Evidence base | Status |
|---|---|---|
| RQ1 | ViperBench coverage (22 kernels × 3 backends × dtype/shape) as a *probe*; the **"passes-but-slow" demonstration** (naive `T.serial` kernels: correct yet 5–314× slow); a **survey of KernelBench** (correctness + speedup-vs-eager, *partial* anti-cheat) and **TritonBench** (Triton-only, fixed shapes); `AKO4ALL/bench/kernelbench/` anti-cheat code as evidence of known reward-hacking modes | **Partly new** — survey + "passes-but-slow" framing are new; the kernels/data exist |
| RQ2 | Existing 22-kernel gap data (`profile.csv`/`slow_kernels.csv`) + NCU root-cause analysis (RC0a authoring / RC0b,RC1 codegen / RC2 autotune / RC3 LN-spill / RC4 Winograd) + fp32→TF32 + cross-arch | **Transfers ~verbatim** (this is `analysis.tex` today) |
| RQ3 | Mitigation campaigns (12 kernels, rounds 1+2) recast as **patterns** (dominant authoring fix `T.serial`→`T.reduce` + native-dtype I/O + `torch.empty`; autotune expansion; implicit-GEMM conv); the **comparability rules + roofline anchor**; the authoring-reducible vs structural-irreducible split | **Transfers as patterns**; rules + roofline validation are new |

## New evidence to collect (experiments)

1. **Benchmark survey (RQ1):** systematic coverage + anti-cheat analysis of KernelBench & TritonBench
   (DSL, dtype, shape, hardware, reward-hacking). Mostly desk/code analysis.
2. **"Passes-but-slow" demonstration (RQ1):** run representative naive kernels through KernelBench-style
   gating; show they pass yet are 5–300× slow. We have the kernels + `AKO4ALL` harness.
3. **Heuristic validation (RQ3):** show the comparability rules + roofline fraction discriminate good
   from poor kernels across the suite (flag the slow, clear the fast); roofline breaks PyTorch-circularity.
4. **GH200 → A100-SXM4 (→ Ada) multi-GPU re-baseline:** breadth for RQ1 hardware-coverage + RQ2.
   *Note:* keep **naive + optimized** variants per kernel and time both (RQ2 gap = naive; RQ3 = optimized;
   the cliff = naive/optimized). Do **not** overwrite naive with optimized.

## The heuristics (RQ3) — define precisely, present with their limits

- **Comparability rules (pragmatic screen):** a DSL kernel should (i) be within ε of the vendor/PyTorch
  baseline on representative shapes, (ii) be at least at parity and ideally faster at large input sizes,
  (iii) show no catastrophic per-shape collapse. *Limit:* PyTorch-relative is **circular** (PyTorch is
  both baseline and de-facto "good") — use only as a screen.
- **Roofline anchor (baseline-independent):** achieved fraction of the memory/compute roofline; ≥ ~X% of
  achievable roofline ⇒ "well-written" regardless of the library. This is the non-circular leg that makes
  the rules defensible. **Required, not optional.**
- **Pattern checklist:** the recurring authoring fixes (reduction primitive, native-dtype I/O, allocation,
  autotune coverage) as a developer-facing debug checklist (drafted already in `discussion.tex`).

## Guardrails (reviewer-proofing — do not violate)

1. **Benchmark critique must be precise and fair.** KernelBench = LLM-gen benchmark (correctness +
   speedup-vs-eager, *partial* anti-cheat); TritonBench = Triton-only. No strawman; reviewers know these.
2. **The comparability rules are circular without the roofline anchor.** Always present them together and
   validate they discriminate.
3. **Do not undercut ViperBench.** It is a *diagnostic probe* that exposes the evaluation gap, **not** a
   proposed comprehensive benchmark (we argue comprehensive is infeasible).
4. **"Passes-but-slow" is evidence about *benchmarks*,** not a claim that 300× kernels are typical in prod.
5. **Do not regress the corrected RC labels** (RC0 memory-latency not barriers; RC3 = TileLang LayerNorm
   spill not conv; RC4 Winograd ≈2–3%). See `CLAUDE.md` "Corrected root-cause labels".

## Carryover engineering items (still valid regardless of framing; from old `REVISION_TODO.md`)

- **N1:** paper said NHWC + `allow_tf32=False` + `cudnn.benchmark=False`; code ran NCHW + defaults.
  Decision (2026-06-25): **fix code + re-run** with `channels_last` + flags on the canonical GPU.
- **Locked-clock benchmarking** standard (needs `sudo`); adopt as the measurement standard.
- **E_lib computed by a committed script** (de-risk hand-typed table numbers); `compute_elib.py` exists.
- Corrected RC-label propagation: **already done** in abstract/intro/analysis/`tab:rootcauses`.

## Doc reconciliation manifest

**UPDATE (living docs → rewrite to new framing):** `paper-latex-project/tex/*.tex` (all sections),
`main.tex` (title), `paper-latex-project/CLAUDE.md`, repo `CLAUDE.md` (framing + RQ map),
`README.md`, `AKO4ALL/README.md` (recast as the LLM-kernel-gen / reward-hacking asset), memory.

**KEEP as historical record (do NOT edit/delete):** `reviews.txt` (raw reviewer text),
`REBUTTAL.md` (what was submitted under the old framing), `ase26-paper4134.pdf` (submitted PDF).
These document the prior review round; deleting them loses the record.

**KEEP as data/results (not framing):** `AKO4ALL/results/optimization_results*.csv`,
`OPTIMIZATION_V2_SUMMARY.md`, `experiments/results/**`, `ViperBench/results/**`.

**RECONCILE then retire:** `REVISION_TODO.md` (live engineering items captured above ↑; rest obsolete).

**DROP (stale, tied to the abandoned old-framing rebuttal; git-recoverable):** `logs/REBUTTAL_GAME_PLAN.md`,
`logs/REBUTTAL_EXPERIMENT_PROTOCOLS.md`, `logs/REBUTTAL_PROTOCOLS_CRITICAL.md`,
`logs/ADDITIONAL_EXPERIMENTS_PLAN.md`, `logs/REVIEWER_WEAKNESS_ANALYSIS.md`.
**KEEP for now (audits with reusable integrity checks):** `logs/CONSISTENCY_AUDIT.md`,
`logs/RIGOR_AUDIT.md`, `logs/rebuttal_audit/*`, `logs/README.md` — re-evaluate after the section rewrites.

## Status / progress

- [x] Canonical framing locked (this file)
- [x] `abstract.tex`, `RQ_summary.tex` rewritten to new framing
- [x] repo `CLAUDE.md` pivot banner
- [x] `introduction.tex` rewrite (thesis-bearing) — DONE; uses `\cite{kernelbench}` (bib entry TODO)
- [x] `evaluation.tex` → §5 retitled "The Evaluation Gap (RQ1)"; prepended benchmark survey + passes-but-slow demo; magnitude tables kept in place reframed as "the hidden gap these benchmarks admit". main.tex §5/§6/§7 retitled. `kernelbench` bib entry added. (Tables NOT moved — labels global, lower risk.)
- [x] `methodology.tex` → added "Evaluation-Gap and Heuristic Measurement" subsection (passes-but-slow harness + roofline essential-work model) and GH200 to the hardware setup.
- [x] `analysis.tex` (RQ2) — added the pivot framing tie (kernels pass benchmarks yet lag; causes split by where the fix belongs: authoring / codegen / library-maturity). RC0-RC4 body already counter-grounded; corrected labels preserved.
- [x] `mitigation.tex` → recast as RQ3 guidance: prepended heuristics subsection (comparability screen + roofline anchor) + `tab:roofline` (GH200 cliff/ρ/E_lib, authoring-artifact vs structural-residual dichotomy, judgment-band honesty); existing campaigns kept as the optimization-pattern evidence.
- [x] `background.tex` → §2.5 recast as "Benchmarking DSL and LLM-Generated Kernels": added KernelBench (correctness-gated, fast-only guard) + LLM-kernel-gen framing alongside TritonBench; sets up the RQ1 evaluation gap.
- [x] `related_work.tex` (eval-gap novelty + KernelBench, TritonBench roofline cited as precedent not novel), `discussion.tex` (benchmark-design implication: gate on performance), `conclusion.tex` (full eval-gap reframe), `threats.tex` (probe-not-benchmark per guardrail 3 + roofline-assumption + GPU-mix threats)
- [~] `main.tex` title — DEFERRED by author decision (keep current title for now); ancillary docs (`README`, `AKO4ALL/README`); drops
- [x] New experiments: benchmark survey (`RQ1_benchmark_survey.md`), passes-but-slow demo (`exp_passes_but_slow.py`), heuristic validation (`exp_cliff_roofline.py` → `tab:roofline`) — all DONE on GH200. GH200 re-baseline: `profile.GH200-480GB.csv` + tuning sweep done; authoritative locked-clock + ncu pending the user's `sudo` pass (`repro/run_pipeline.sh`).

> **2026-06-26: ALL 12 `.tex` sections pivoted** (abstract, intro, background, methodology, RQ_summary, evaluation/RQ1, analysis/RQ2, mitigation/RQ3, discussion, related, threats, conclusion) + main.tex titles. Repo-wide scan: every `\cite`/`\cref` resolves. No LaTeX toolchain on this box — verified structurally; **build the PDF on an author machine / Overleaf**. Open follow-ups: (a) kernelbench arXiv id verify (`references.bib` TODO); (b) A100-suite tables vs GH200 demo numbers — the in-progress GH200 re-baseline will unify (TODOs left inline); (c) ancillary docs (`README.md`, `AKO4ALL/README.md`) not yet recast; (d) title still deferred.
