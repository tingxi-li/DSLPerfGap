# CLAUDE.md — ASE 2026 Paper #4134 (LaTeX source)

## Project Overview

> ⚠️ **PAPER PIVOT (2026-06-25).** The paper is being completely re-framed around **the DSL kernel
> *evaluation* problem** (existing benchmarks pass correct-but-slow kernels; a comprehensive benchmark
> is infeasible; we offer evaluation heuristics + optimization patterns). The canonical new thesis, RQ
> wording, candidate titles, and section-rewrite checklist live in **`../PIVOT_FRAMING.md`** — read it
> before editing any `.tex`. The "Research topic", "Core thesis", and "Paper Structure & Research
> Questions" sections below describe the **OLD** framing and are being reconciled; new RQs are
> **RQ1 = evaluation gap**, **RQ2 = the hidden gap + causes**, **RQ3 = guidance (heuristics + patterns)**.
> Where this file conflicts with `../PIVOT_FRAMING.md`, the framing doc wins. `abstract.tex` and
> `RQ_summary.tex` are already rewritten; the other sections are pending.

LaTeX source for the ASE 2026 submission "An Empirical Study of GPU Kernel Performance Gaps in Modern Domain-Specific Languages" (paper #4134). The build is `\documentclass[10pt,conference]{IEEEtran}` (see `main.tex:1`); anonymization for review is manual via the `\author` block (`Anonymous Author(s)` / `Anonymous Institution`), not a class option. The submitted PDF and reviewer text live one directory up at `../ase26-paper4134.pdf` and `../reviews.txt`.

**Research topic:** An empirical study on the performance gap between modern Domain-Specific Languages (DSLs) — specifically Triton and TileLang — and PyTorch + cuBLAS/cuDNN, with root-cause analysis and proposed mitigations.

**Core thesis:** DSLs like Triton and TileLang outperform PyTorch on some GEMM/element-wise shapes but underperform on convolution and (for TileLang) on reductions/normalization. This paper provides: (1) a benchmark evaluation across a 22-kernel suite, (2) root-cause analysis grounded in Nsight Compute counters, and (3) measured mitigations recovering up to 1224× on TileLang LayerNorm and reaching ≥95% library efficiency on 4/5 fixed kernels.

> **Parent-repo orientation:** This directory is one subsystem of `../`. The parent `../CLAUDE.md` contains the section/table/figure → code/data mapping, the corrected RC label callout (RC0/RC3/RC4 attributions changed between submission and revision — do **not** re-introduce the submitted wording), and pointers to every data source. Read it before drafting analysis/mitigation prose.

---

## Repository Structure

```
.
├── CLAUDE.md                                # This file
├── main.tex                                 # Top-level entry; \documentclass[10pt,conference]{IEEEtran}
├── references.bib                           # BibTeX bibliography (loaded at main.tex:132)
├── acmart.cls / ACM-Reference-Format.bst    # UNUSED leftover ACM files; build uses IEEEtran from the TeX distribution
├── main.pdf                                 # Built output (commit-tracked)
├── known_github_issues.md                   # Background material (Triton/TileLang upstream issues)
├── tex/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── background.tex          # Tile-based GPU programming, Triton, TileLang, cuBLAS/cuDNN, TritonBench
│   ├── methodology.tex         # 22-kernel suite, baselines, profiling setup, metrics, hardware
│   ├── RQ_summary.tex          # RQ1/RQ2/RQ3 definitions (\input'd between methodology & evaluation)
│   ├── evaluation.tex          # RQ1: tab:gemm, tab:conv, tab:norm, tab:summary, fig:overview
│   ├── analysis.tex            # RQ2: RC0–RC4 narrative + tab:rootcauses
│   ├── mitigation.tex          # RQ3: tab:mitig:norm, tab:mitig:conv, tab:mitigation
│   ├── discussion.tex
│   ├── related_work.tex
│   ├── threats.tex             # Threats to Validity
│   └── conclusion.tex
├── figures/
│   ├── overview_efficiency.pdf # The ONLY figure embedded in the built doc (fig:overview)
│   └── gen_overview.py         # Generator script — DATA HARD-CODED from tex/evaluation.tex tables
└── docs/
    └── superpowers/plans/      # Author-side planning markdowns (not built)
```

> **Note on data sources.** The paper folder keeps **no local copies** of benchmark CSVs or analysis docs. The former `profile.csv`, `slow_kernels.csv`, and `GPU Kernel Performance Analysis Report.md` copies (plus the pre-pivot `PROGRESS.md` tracker) were removed on 2026-06-29 after going stale against the dual-arch pivot — they were single-arch RTX 4000 Ada data / old-framing. All table numbers are hand-typed from the source-of-truth files in `../ViperBench/results/` (now per-architecture: `profile.A100-SXM4-40GB.csv`, `profile.GH200-480GB.csv`, plus `slow_kernels.csv`) and `../experiments/results/<gpu>/`; consult those directly.

---

## Build Instructions

```bash
# Full build (run twice for cross-references)
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Quick draft build (no bibliography)
pdflatex main.tex

# Clean auxiliary files
rm -f *.aux *.bbl *.blg *.log *.out *.toc
```

Current build uses `\documentclass[10pt,conference]{IEEEtran}`. Anonymization for review is manual via the `\author` block (`Anonymous Author(s)` / `Anonymous Institution`); there are no class-option venue toggles to switch for camera-ready.

---

## Paper Structure & Research Questions

The paper is organized around three RQs (defined verbatim in `tex/RQ_summary.tex`):

- **RQ1 (Characterization), `tex/RQ_summary.tex:20`:** "What performance gap separates Triton and TileLang kernels from cuBLAS and cuDNN baselines, and how does this gap differ across operator categories?"
- **RQ2 (Root-Cause Analysis), `tex/RQ_summary.tex:22`:** "What compiler and architectural factors contribute most to the observed gaps?"
- **RQ3 (Mitigation), `tex/RQ_summary.tex:24`:** "How much performance can be recovered by targeted implementation fixes derived from the root-cause analysis?"

Each RQ is owned by one section: RQ1 → `tex/evaluation.tex`, RQ2 → `tex/analysis.tex`, RQ3 → `tex/mitigation.tex`. Tables and figures inside each section are listed in the parent repo's `../CLAUDE.md` under "Paper ↔ artifact mapping".

---

## Writing Guidelines for Claude

### Voice & Style
- Academic, precise, and direct. Avoid filler phrases ("it is worth noting that", "in order to").
- Passive voice is acceptable for methods sections; prefer active voice in introduction and discussion.
- Use present tense for describing the paper's contributions; past tense for describing experiments.
- Avoid over-hedging — state findings confidently while acknowledging limitations in the Discussion.

### Terminology (use consistently)
| Preferred term | Avoid |
|---|---|
| DSL / domain-specific language | "custom language", "scripting language" |
| Triton | "OpenAI Triton" (unless first mention) |
| TileLang | "tilelang" (capitalize as shown) |
| cuBLAS / cuDNN | "CUDA library", "cuda lib" |
| kernel | "function", "program" (in GPU context) |
| throughput (TFLOPS) | "speed", "performance" (be specific) |
| TritonBench | "the benchmark", "the suite" (after first mention) |

### Section-specific notes
- **Introduction (`tex/introduction.tex`):** Practitioner motivation (GPU kernel dev moving to DSLs), explicit RQ summary, bulleted contributions. Already complete.
- **Background (`tex/background.tex`):** Tile-based GPU programming, Triton, TileLang, cuBLAS/cuDNN, TritonBench. SE audience — assume compiler literacy but not GPU literacy.
- **Methodology (`tex/methodology.tex`):** Pinned hardware string (RTX 4000 Ada / sm_89, CUDA 12.6, Driver 595) at `tex/methodology.tex:106-107`; pinned software (PyTorch 2.8.0, Triton 3.4.0, TileLang 0.1.6.post1, cuDNN 9.1.0.2, Nsight Compute 2024.3.2.0) at L110. Timing uses `cudaEventRecord`/`cudaEventElapsedTime`. **Note:** the abstract and methodology say **22 kernels** but `tex/introduction.tex:48` currently says **21** — fix before camera-ready.
- **Evaluation (`tex/evaluation.tex`):** Leads each finding with a bold paragraph and `\phantomsection\label{...}` instead of `\subsection`. RQ1 data: `tab:gemm`, `tab:conv`, `tab:norm`, `tab:summary`, `fig:overview`. Numbers are hand-typed from `../ViperBench/results/slow_kernels.csv` and `profile.csv`.
- **Analysis (`tex/analysis.tex`):** RC0–RC4 + `tab:rootcauses`. Grounded in `../experiments/results/<gpu>/NCU_FINDINGS.md` and `ncu_summary.csv`. **Read the corrected-RC-labels callout in `../CLAUDE.md` before editing.**
- **Mitigation (`tex/mitigation.tex`):** Five fixed kernels (`tab:mitigation`) feed from `../AKO4ALL/results/optimization_results.csv` rows. Conv arm also uses `../experiments/results/<gpu>/conv_mitigation*.csv`. PyTorch baseline footnote at `tab:mitig:conv` (≈9% drift from `tab:conv`) is intentional and documents clock variation across runs. **Dual-arch (2026-06-26):** `tab:mitigation` now has a GH200 `E_lib` column (from `../experiments/results/NVIDIA_GH200_480GB/cliff_roofline.csv`, = `tab:roofline`); the GH200 norm/reduction recovery is scoped to exclude `logsumexp` (sm_90 RC3 spill) — see Out of Scope.
- **Threats to Validity (`tex/threats.tex`):** mentions `cudnnFind` and `torch.backends.cudnn.benchmark` reproducibility settings.

---

## LaTeX Conventions

- One sentence per line in `.tex` source (aids version control diffs).
- Use `\cref{}` (cleveref) for all cross-references, not `Figure~\ref{}`.
- Tables: use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`); no vertical rules.
- Figures: vector PDF preferred; `\includegraphics[width=\linewidth]{figures/foo.pdf}`.
- Highlight best results in tables with `\textbf{}`.
- Use `\etal` macro or `et al\onedot` consistently for citations.
- Keep `references.bib` entries clean: no URLs in `title`, proper capitalization with `{}` guards (e.g., `{GPU}`, `{Triton}`).

---

## Data sources for paper claims (read before editing tables/figures)

Every number in the paper is hand-typed from a CSV in the parent repo. Re-deriving a claim means going back to the CSV; the build does NOT regenerate numbers automatically.

| Paper artifact | Source file |
|---|---|
| `tab:gemm` numbers | `../ViperBench/results/slow_kernels.csv` (matmul/batched_matmul/linear_activation, large) + `../ViperBench/results/profile.csv` (small/medium configs) |
| `tab:conv` numbers | `../ViperBench/results/slow_kernels.csv` (conv2d large) + `../experiments/results/<gpu>/conv_filters_small.csv` (small Conv2d 8×64×56×56) |
| `tab:norm` numbers | `../ViperBench/results/slow_kernels.csv` (layer_norm, rms_norm rows) |
| `tab:summary` | Hand-derived medians from `tab:gemm`/`tab:conv`/`tab:norm` |
| `tab:rootcauses` | `../experiments/results/<gpu>/ncu_summary.csv` + `NCU_FINDINGS.md` |
| `tab:mitig:norm` rows | `../AKO4ALL/results/optimization_results.csv` (rows: layer_norm tilelang, rms_norm tilelang) |
| `tab:mitig:conv` row | `../AKO4ALL/results/optimization_results.csv` (row: conv2d triton) + `../experiments/results/<gpu>/conv_mitigation*.csv` |
| `tab:mitigation` | Aggregates 5 rows of `../AKO4ALL/results/optimization_results.csv` |
| `fig:overview` (the only figure) | `figures/overview_efficiency.pdf`, generated by `figures/gen_overview.py` with **hard-coded** arrays (docstring: "Data hard-coded from evaluation tables in tex/evaluation.tex.") |

Full mapping (sections + RQs + figures) lives in `../CLAUDE.md` under "Paper ↔ artifact mapping".

The parent-repo `REVISION_TODO.md` and `experiments/results/<gpu>/NCU_FINDINGS.md` document three RC re-attributions (RC0 = memory latency not barriers; RC3 = TileLang-LN spill not conv; RC4 = Winograd contributes only ~2–3%). The submitted PDF wording is wrong on all three — do not regress.

---

## Key References (seed list, in `references.bib`)

- TritonBench (source of kernel suite)
- Triton language paper (Tillet et al.)
- TileLang (Wang et al.)
- cuBLAS / cuDNN documentation
- Triton GitHub discussion #591 (conv performance gap)
- FlashAttention-2 (attention baseline)
- Relevant DSL/GPU compiler papers (Halide, TVM, MLIR)

---

## Out of Scope for Claude

- Do not modify or generate profiling scripts, benchmark code, or kernel implementations under `../ViperBench/`, `../AKO4ALL/`, or `../experiments/`.
- Do not invent experimental results, numbers, or performance figures. If a number is not in one of the CSVs listed under "Data sources", flag it with `% TODO: verify with data` rather than guessing.
- Do not change the ACM document class, anonymization flags, or formatting macros without being asked.
- Do not re-introduce the submitted PDF's RC0/RC3/RC4 wording (see `../CLAUDE.md` "Corrected root-cause labels" and `../REVISION_TODO.md`).
- Do not re-introduce an unqualified "entire normalization and reduction family recovers" claim. On the **GH200** the family recovers *except* `logsumexp`, whose optimized kernel register-spills on sm_90 (RC3-class; ~280 GB local spill / 255 regs — captured in `../experiments/results/NVIDIA_GH200_480GB/ncu_summary.csv`). The abstract, conclusion, `mitigation.tex`, and `evaluation.tex` are deliberately scoped to "every family kernel but `logsumexp`" — keep the caveat. The A100 `tab:mitigation` LogSumExp 581.7% carries a `% TODO(verify)` (the same kernel spills on sm_90; confirm sm_80 does not before trusting it) — leave it. Background: `../CLAUDE.md` AKO section + parent memory `logsumexp-gh200-nontransfer`.
- The five suite + mitigation tables are **dual-arch (A100 + GH200)** as of 2026-06-26; GH200 suite numbers are the *unlocked tuned* profile (do not locked-re-baseline the GH200 suite — it would mismatch the A100 methodology). See `../CLAUDE.md` "Tables → data" dual-arch note.