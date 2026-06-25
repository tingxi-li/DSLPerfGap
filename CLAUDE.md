# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> ⚠️ **PAPER PIVOT (2026-06-25) — read `PIVOT_FRAMING.md` first.** The paper has pivoted from
> *"An Empirical Study of GPU Kernel Performance Gaps in Modern DSLs"* (characterize → root-cause →
> mitigate a gap) to **the DSL kernel *evaluation* problem**: existing benchmarks (KernelBench,
> TritonBench) pass correct-but-slow kernels, a comprehensive benchmark is infeasible, and we propose
> evaluation heuristics + optimization patterns to guide DSL kernel development. **`PIVOT_FRAMING.md`
> (repo root) is the single source of truth for the new thesis and RQs.** The old gap study is now
> *evidence* (it became RQ2), not the thesis. The RQ labels below describe the OLD framing and are being
> reconciled section-by-section; where any doc conflicts with `PIVOT_FRAMING.md`, the framing doc wins.
> New RQ map: **RQ1 = evaluation gap (benchmarks miss slow kernels)**, **RQ2 = the hidden gap + its
> causes** (the old RQ1+RQ2 evidence), **RQ3 = guidance heuristics + optimization patterns**.

## What this repository is

Research artifact for ASE 2026 paper #4134 (`ase26-paper4134.pdf`, reviewer text in `reviews.txt`, author response in `REBUTTAL.md`, in-flight revision items in `REVISION_TODO.md`). Five coupled subsystems:

- **ViperBench/** — benchmark suite comparing **PyTorch**, **Triton**, and **TileLang** on 22 deep-learning kernels for correctness and latency/memory. This is the **RQ1** evidence base.
- **AKO4ALL/** — agentic kernel-optimization loop: a coding agent (Claude Code) iteratively rewrites a single kernel for maximum speed under a strict protocol. Source of the **RQ3** mitigation campaigns.
- **experiments/** — post-submission rebuttal evidence: Nsight Compute counter sweeps, FP32 GEMM root-cause, Winograd isolation, clock-locked re-timing, fused/eager baseline split, conv filter coverage. Results namespace per GPU (`results/<gpu_slug>/...`). This is the **RQ2** evidence base.
- **logs/** — rebuttal planning, audit, and reviewer-weakness docs (working notes, not built into the paper).
- **paper-latex-project/** — the LaTeX paper draft (`main.tex` + `tex/*.tex`, built `main.pdf`). Has its own `CLAUDE.md` with writing-style rules; see "Paper draft" section below for the section/table/figure → artifact mapping.

The bridge between ViperBench and AKO4ALL is `AKO4ALL/prepare_kernel.py`, which exports a ViperBench kernel into AKO4ALL's KernelBench format.

> Best starting points: `README.md` (numbered list of rebuttal deliverables with evidence pointers), `AKO4ALL/README.md` (the optimizer), `experiments/A100_H100_RUNBOOK.md` (how to replay the rebuttal suite on a different GPU), `paper-latex-project/main.tex` (the paper itself).

## ⚠️ Stale root-level scaffolding — ignore it

The repo root contains an **abandoned earlier scaffold** that does *not* run against the real project: `run_all.py`, `profile_all.py`, `test_harness.py`, `test_kernel.py`, `tilelang_impl.py`, `pytorch_ref.py`, `bench_gemm_quick.py`, and `tests/results/`. These hardcode `KERNELS_DIR = Path("newBench")` — **a directory that does not exist**. Do not run, edit, or copy patterns from them. The live equivalents all live under `ViperBench/` (e.g. the real runner is `ViperBench/run_all.py`, the real harness is `ViperBench/test_utils.py`). Treat the root scaffold as dead code unless a task explicitly targets it.

## ⚠️ Corrected root-cause (RC) labels — the paper PDF is out of date here

The submitted PDF (`ase26-paper4134.pdf`) used three RC attributions that the rebuttal experiments **disproved**. The corrected attributions are in `REVISION_TODO.md` and `experiments/results/NVIDIA_RTX_4000_Ada_Generation/NCU_FINDINGS.md`. When editing `paper-latex-project/tex/analysis.tex` / `tex/mitigation.tex` / `tex/RQ_summary.tex`, do **not** re-introduce the old framing:

| RC | Submitted-PDF claim (wrong) | Corrected attribution (use this) |
|----|------------------------------|----------------------------------|
| RC0 | TileLang `T.serial` reductions stall on barrier sync | Memory-latency stalls, not barriers — `T.reduce` recovers it via better scheduling |
| RC3 | Convolution suffers from register spill | Spill is **TileLang-LayerNorm-specific**, not a Triton-conv problem |
| RC4 | "Winograd primarily" explains the conv gap | Winograd contributes only ~2–3%; soften this claim |

These are tracked as TODO items in `REVISION_TODO.md` with PDF line pointers.

## Common commands

```bash
pip install torch triton tilelang        # ncu (Nsight Compute) also needed for AKO4ALL profiling

# Correctness
python ViperBench/run_all.py              # run every ViperBench/<kernel>/test.py, print summary
python ViperBench/<kernel>/test.py        # one kernel (e.g. layer_norm); exits 0 pass / 1 fail

# Latency + peak-memory benchmarks → ViperBench/results/profile.csv
python ViperBench/benchmark.py            # PyTorch + Triton (+ *_tuned variants)
python ViperBench/benchmark_tilelang.py   # TileLang
python ViperBench/benchmark_tuned.py      # all impls including tuned

# Auto-tuning sweep (writes results/tuning_cache.json) — run as a module from inside ViperBench/
cd ViperBench && python -m tuning.sweep --all
cd ViperBench && python -m tuning.sweep --kernel matmul --impl triton

# AKO4ALL: optimize one kernel
cd AKO4ALL && python prepare_kernel.py <kernel> <triton|tilelang>   # export from ViperBench
bash scripts/bench.sh baseline            # verify CORRECT=True before optimizing
cd AKO4ALL && claude                      # then: "Follow the instructions in TASK.md."

# Rebuttal experiments (per-GPU namespaced under experiments/results/<gpu_slug>/)
bash experiments/run_all.sh                                 # serial master runner; pins one GPU
python experiments/exp_fp32_gemm.py                         # RC FP32 TF32-lowering root cause
python experiments/exp_conv_filters.py                      # conv 1×1/3×3/5×5/7×7 + depthwise sweep
python experiments/exp_fused_baselines.py                   # eager vs torch.compile vs DSL baseline split
python experiments/exp_autotune_matmul.py                   # reconciles §5 vs §7.3 matmul autotune
python experiments/exp_winograd_isolation.py                # RC4: Winograd-eligible vs not
python experiments/exp_correctness_edge.py                  # NaN/Inf/denormal edge inputs
python experiments/exp_significance.py                      # clock-locked re-timing (needs nvidia-smi -lgc)
bash experiments/ncu_counters.sh <kernel> <impl> <size>     # Nsight Compute counters per target
python experiments/consolidate_ncu.py                       # roll up ncu CSVs → ncu_summary.csv

# Paper LaTeX (from inside paper-latex-project/)
cd paper-latex-project && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

There is no build step or linter; "tests" means the per-kernel correctness scripts above.

## ViperBench architecture

Each kernel lives in `ViperBench/<kernel>/` with exactly four files:

| File | Role |
|------|------|
| `pytorch_impl.py` | reference (torch / cuDNN built-ins) — the correctness golden |
| `triton_impl.py`  | custom Triton kernel |
| `tilelang_impl.py`| custom TileLang kernel |
| `test.py`         | wires all three into the shared harness and `sys.exit()`s |

**Unified-API contract (the key invariant).** All three `*_impl.py` files export a function with the **same name as the kernel directory** and the **same signature**, so they are drop-in interchangeable (e.g. every backend defines `layer_norm(x, weight, bias, eps=1e-5)`). The PyTorch reference deliberately *raises* on argument values the hand-written kernels don't support (e.g. `layer_norm` rejects `eps != 1e-5` because the Triton kernel hardcodes it) — this keeps the three backends locked to identical behavior. **If you add or change a kernel, all three signatures must stay aligned or tests break.**

**Test harness** (`ViperBench/test_utils.py`):
- `run_test(...)` compares PyTorch vs Triton; `run_tilelang_test(...)` compares PyTorch vs TileLang. `test.py` picks one. Both write JSON to `ViperBench/results/<kernel>[_tilelang].json` and call `sys.exit(0|1)` — so `test.py` runs on import; always invoke it as a subprocess, never import it.
- Tolerances by dtype: fp32 `1e-5`, fp16 `1e-3`, bf16 `1e-2`. Pass `loose_tol=True` (used for reductions / normalizations) to double them for order-of-operations drift.
- A `test.py` defines `test_cases` (list of `{"name", "inputs", "dtype"}`); `inputs` may be a tuple/list (positional) or dict (keyword).

**Tuning is loaded at import time — and silently arch-dependent.** Every `*_impl.py` begins with:

```python
from tuning.cache import get_best_config as _get_best_config
_TUNED = _get_best_config("<kernel>", "<impl>") or {}   # falls back to {} on any error
```

then reads block sizes / thread counts / num_stages from `_TUNED` with hardcoded defaults (`_TUNED.get("BLOCK_SIZE_M", 64)`). `cache.py` keys configs as `"<kernel>/<impl>/<gpu_arch>"` in `ViperBench/results/tuning_cache.json`, where `gpu_arch` comes from `torch.cuda.get_device_name(0)`. **Consequence:** on a GPU/arch with no cached entry you transparently get the defaults — kernel correctness never changes, but performance does. The candidate grids per kernel live in `ViperBench/tuning/configs.py` (`TRITON_CONFIGS`, `TILELANG_CONFIGS`); `sweep.py` times them and writes the winners back to the cache.

**Benchmarking.** `benchmark*.py` import each impl via `importlib`, run CUDA-synchronized `perf_counter` timing (10 warmup / 100 measured, median) plus `torch.cuda.max_memory_allocated()`, and append rows to `results/profile.csv` with columns `kernel,size,impl,input_desc,latency_ms,peak_memory_mb`. `impl` ∈ {`pytorch`, `triton`, `triton_tuned`, `tilelang`, `tilelang_tuned`}. Input shapes come from `kernel_input_shapes.html` (each `benchmark*.py` reproduces the spec in `get_test_cases()`). `results/slow_kernels.csv` is the derived "DSL slower than PyTorch at large" report consumed by the paper's headline tables. `benchmark_attn.py` patches the attention-large row (reduced `D=64` to dodge the Triton shared-memory limit); `benchmark_fix.py` re-patches three rows after correctness fixes — both edit `profile.csv` in place rather than producing new artifacts.

## AKO4ALL architecture

`prepare_kernel.py <kernel> <impl>` is the ViperBench→AKO4ALL bridge: it reads `ViperBench/<kernel>/pytorch_impl.py` (golden) and `<impl>_impl.py` (target), and emits `input/reference.py` + `solution/kernel.py` in **KernelBench format** plus `scripts/bench.sh`. Per-kernel large-input shapes are hardcoded in its `KERNEL_CONFIGS` dict (only kernels listed there can be exported this way).

The optimization loop is governed by two files a running agent must obey **exactly**:
- `TASK.md` — rigid protocol: analyze `input/`+`context/`+`bench/`+`HINTS.md` → create an `opt/<kernel>` git branch → copy kernel to `solution/` → generate `scripts/bench.sh` (fill `{{BENCH_COMMAND}}` in `bench-wrapper.sh`) → verify baseline `CORRECT=True` and commit. Then iterate: **one iteration = one code edit + one `bash scripts/bench.sh iter-N` run**; after each, update `ITERATIONS.md` and `git commit -m "[iter N] ..."`. Goal is *genuine* latency reduction — reward hacking (stream injection, timing monkey-patching, uninitialized output) is forbidden.
- `HINTS.md` — constraints: run `ncu` before iteration 1 and re-profile after 3 non-improving iterations; optimize for large inputs; **never switch language** (TileLang stays TileLang, Triton stays Triton — no PyTorch/cuDNN fallback); do not install packages.

Other pieces:
- `bench/kernelbench/` — built-in correctness+timing evaluator (used when `bench/` has nothing else). Anti-cheat: warns on >10× speedup and overrides the solution's `get_inputs`/`get_init_inputs` with the reference's.
- `context/` — reference docs the agent may consult: `GPU Kernel Performance Analysis Report.md`, `tilelang_reference.md`, `triton_tuning.md`, `known_github_issues.md`.
- `results/optimization_results.csv` + `results/optimized/<kernel>_<impl>.py` + `<kernel>_iterations.md` — the **five completed campaigns** that feed `tab:mitigation` in the paper:
  1. `layer_norm` TileLang BF16 — 1090 → 0.89 ms (**1224×**, RC0a)
  2. `rms_norm` TileLang FP16 — 716 → 0.90 ms (**796×**, RC0a)
  3. `argmax` TileLang FP16 — 16.2 → 1.75 ms (**9.26×**, RC0a)
  4. `conv2d` Triton FP16 — 32.1 → 12.5 ms (**2.57×**, RC1 + RC2)
  5. `matmul` Triton FP16 — 2.72 → 1.63 ms (**1.66×**, RC2a + L2 swizzle)

  Dominant TileLang win pattern: replace `T.serial` reduction loops with `T.reduce`, do native-dtype I/O, use `torch.empty` over `torch.zeros`. `optimization_results.csv` schema: `kernel,dsl,precision,input_desc,pytorch_ms,before_ms,before_vs_pytorch,after_ms,after_vs_pytorch,speedup_gained,iterations,strategy`.
- `.gitignore` excludes per-run AKO4ALL artifacts: `scripts/`, `trajectory/`, `_bench_output.txt`, `.claude/`. Note the artifact root itself is **not** a git repo — the AKO4ALL workflow initializes/uses git only within its own run.

## Post-submission rebuttal experiments (`experiments/`)

This directory is the **rebuttal evidence base** (RQ2 root-cause, plus correctness/significance follow-ups). It is GPU-portable: every output is namespaced under `experiments/results/<gpu_slug>/` via `_harness.device_slug()`, so the same code replays on A100/H100 without overwriting Ada results (`A100_H100_RUNBOOK.md` documents the replay procedure).

| Script | Reviewer ask | Output(s) in `experiments/results/<gpu>/` |
|--------|--------------|--------------------------------------------|
| `_harness.py` | shared lib (`device_info`, `device_slug`, `time_kernel`, `load_impl`, `write_csv`) | — |
| `exp_fp32_gemm.py` | R1-Q1 / W1 / W11 — TileLang FP32 silent TF32 lowering | `fp32_gemm.csv` |
| `exp_conv_filters.py` | W6 / W13 — 1×1/3×3/5×5/7×7 + depthwise sweep, `n_regs`/`n_spills` | `conv_filters{,_small,_large}.csv` + `conv_mitigation*.csv` |
| `exp_fused_baselines.py` | W2 / R2 / R3 — eager vs `torch.compile(max-autotune)` vs DSL | `fused_baselines.csv` |
| `exp_autotune_matmul.py` | W7 — reconcile §5 (Δ≈0pp @ 16384²) vs §7.3 (1.66× @ 4096²) | `autotune_matmul.csv` |
| `exp_winograd_isolation.py` | RC4 — Winograd-eligible vs not | `winograd_isolation.csv` + `cudnn_winograd_3x3.log` |
| `exp_correctness_edge.py` | R3 / W11 — NaN/Inf/denormal/all-equal edge inputs | `correctness_edge.csv` |
| `exp_significance.py` | R1 — clock-locked re-timing near-parity kernels | `significance.csv`, `significance_smoke.csv`, `clock_lock.txt` |
| `ncu_counters.sh` (drives `run_one_kernel.py`) | R1 — Nsight Compute counter sweep | `ncu/<kernel>_<impl>_<size>_{loads,regs,stalls,l2dram}.csv` |
| `consolidate_ncu.py` | roll-up | `ncu_summary.csv` + stdout RC table |
| `run_all.sh` | serial master runner | `run_all.log` |

The canonical RC interpretation of the ncu sweep is `experiments/results/<gpu>/NCU_FINDINGS.md` (cited by `README.md`, `REBUTTAL.md`, and the paper's `tab:rootcauses`). `exp_autotune_matmul.py` imports `AKO4ALL/results/optimized/matmul_triton.py` directly as the §7.3 "expanded autotune" arm — so the two subsystems are wired, not just textually cross-referenced.

## Rebuttal docs (`logs/`, `REBUTTAL.md`, `REVISION_TODO.md`)

- `REBUTTAL.md` — canonical author response. §1 reviewer-grouped response, §2 concern→action mapping, §3 detailed explanation/integrity stance, §4 open revision items. **Single source of truth** for what was promised to reviewers.
- `REVISION_TODO.md` — author-decision checklist for the revision round: every item ties to a PDF line and a corrected-RC attribution (see "Corrected root-cause labels" callout above).
- `reviews.txt` — raw reviewer text (4134A/B/C); ~187 lines, cited by everything in `logs/`.
- `logs/` — working notes (NOT built into the paper):
  - `ADDITIONAL_EXPERIMENTS_PLAN.md` — master plan mapping each experiment to reviewer asks (W1–W13); records that the suite is Ada-complete and that RC0/RC3/RC4 attributions were corrected.
  - `REVIEWER_WEAKNESS_ANALYSIS.md` — per-weakness severity + disposition.
  - `REBUTTAL_GAME_PLAN.md`, `REBUTTAL_EXPERIMENT_PROTOCOLS.md`, `REBUTTAL_PROTOCOLS_CRITICAL.md` — strategy and per-experiment protocols.
  - `CONSISTENCY_AUDIT.md`, `RIGOR_AUDIT.md` — pre-submission multi-agent audits; produced the open items now in `REVISION_TODO.md`.
  - `logs/rebuttal_audit/` — independent 4-agent fact-check of `REBUTTAL.md` (`A_artifact_factcheck.md`, `B_reviewer_concerns.md`, `C_inline_and_census.md`, `D_meta_audit.md`, capstone `00_SUMMARY_AND_EDIT_LOG.md`).
  - `logs/README.md` — folder index.

## Paper draft (`paper-latex-project/`)

The LaTeX source for the paper. Has its own `CLAUDE.md` with writing-style/voice/terminology rules; consult it before editing prose. Quick orientation:

- `main.tex` — entry point; venue `acmart[sigconf,review,anonymous]`, title "An Empirical Study of GPU Kernel Performance Gaps in Modern Domain-Specific Languages".
- `tex/` — 12 section bodies, all `\input{}`-ed from `main.tex` in this order: abstract, introduction, background, methodology, RQ_summary (RQ1/2/3 definitions), evaluation, analysis, mitigation, discussion, related_work, threats, conclusion.
- `figures/` — only **one** figure environment in the built doc (`fig:overview` → `figures/overview_efficiency.pdf`); generated by `figures/gen_overview.py` from **hard-coded numbers** (the script's docstring explicitly says so). Numbers originate in `ViperBench/results/slow_kernels.csv` but were hand-typed into the eval tables and then into the script — there is no automated pipeline from CSV → figure.
- `profile.csv`, `slow_kernels.csv`, `GPU Kernel Performance Analysis Report.md`, `known_github_issues.md` inside `paper-latex-project/` are **duplicates** of files in the parent repo (`ViperBench/results/` and `AKO4ALL/context/`) — copied in for self-contained-build convenience, not regenerated. If you change the parent file, the in-paper copy goes stale.
- `references.bib` — bibliography (loaded at `main.tex:163`). **Not** named `paper.bib` — the paper's own `CLAUDE.md` historically claimed otherwise.

## Paper ↔ artifact mapping

Use this when a paper claim/table/figure needs to be re-derived or re-verified. Paths under `paper-latex-project/` are relative to that directory; everything else is repo-root-relative.

### Sections → primary code/data

| Paper section | LaTeX source | Code / data origin |
|---|---|---|
| §2 Background | `tex/background.tex` | (no code; cites TritonBench, Triton, TileLang) |
| §3 Methodology — Kernel Suite | `tex/methodology.tex` (§3.1) | `ViperBench/<22 kernels>/{pytorch,triton,tilelang}_impl.py`, `kernel_input_shapes.html` |
| §3 Methodology — Profiling | `tex/methodology.tex` (§3.3) | `ViperBench/test_utils.py` (`run_test`, `run_tilelang_test`), `ViperBench/benchmark*.py` |
| §4 Research Questions | `tex/RQ_summary.tex` | — (definitions only) |
| §5 Evaluation (RQ1) | `tex/evaluation.tex` | `ViperBench/results/profile.csv` + `slow_kernels.csv`; tuning effect from `ViperBench/results/tuning_cache.json` |
| §6 Analysis (RQ2, RC0–RC4) | `tex/analysis.tex` | `experiments/results/<gpu>/NCU_FINDINGS.md` + `ncu_summary.csv` + `ncu/*.csv`; `experiments/results/<gpu>/fp32_gemm.csv`; `experiments/results/<gpu>/winograd_isolation.csv`; `experiments/results/<gpu>/autotune_matmul.csv` |
| §7 Mitigation (RQ3) | `tex/mitigation.tex` | `AKO4ALL/results/optimization_results.csv` (5 rows) + `AKO4ALL/results/optimized/<kernel>_<impl>.py` + `<kernel>_iterations.md`; `experiments/results/<gpu>/conv_mitigation*.csv` for the Triton conv arm |
| §8 Discussion | `tex/discussion.tex` | (synthesis; no new data) |
| §9 Related Work | `tex/related_work.tex` | `references.bib` |
| §10 Threats to Validity | `tex/threats.tex` | — |

### Tables → data

| Label | Location | Source data |
|---|---|---|
| `tab:gemm` | `tex/evaluation.tex:295` | `ViperBench/results/slow_kernels.csv` rows for `matmul`, `batched_matmul`, `linear_activation` (large); small-config rows are in `ViperBench/results/profile.csv`. Hand-typed into LaTeX. |
| `tab:conv` | `tex/evaluation.tex:345` | `slow_kernels.csv` row for `conv2d` (large); the small Conv2d $8\times64\times56\times56$ row comes from `experiments/results/<gpu>/conv_filters_small.csv`. |
| `tab:norm` | `tex/evaluation.tex:382` | `slow_kernels.csv` rows for `layer_norm` and `rms_norm`. |
| `tab:summary` | `tex/evaluation.tex:416` | Hand-computed medians over the per-category cells in `tab:gemm`/`tab:conv`/`tab:norm`. |
| `tab:rootcauses` | `tex/analysis.tex:368` | `experiments/results/<gpu>/ncu_summary.csv` + `NCU_FINDINGS.md`; "measured contribution" column cross-cites the mitigation tables. |
| `tab:mitig:norm` | `tex/mitigation.tex:50` | `AKO4ALL/results/optimization_results.csv` rows for `layer_norm tilelang` and `rms_norm tilelang`; full trajectory in `results/optimized/{layer_norm,rms_norm}_iterations.md`. |
| `tab:mitig:conv` | `tex/mitigation.tex:96` | `AKO4ALL/results/optimization_results.csv` row for `conv2d triton` + `experiments/results/<gpu>/conv_mitigation*.csv`; trajectory in `results/optimized/conv2d_iterations.md`. |
| `tab:mitigation` | `tex/mitigation.tex:144` | Aggregates all 5 rows of `AKO4ALL/results/optimization_results.csv`. |

### Figures → data

| Label | Location | Source data |
|---|---|---|
| `fig:overview` | `tex/evaluation.tex:259` → `figures/overview_efficiency.pdf` | Generated by `figures/gen_overview.py` from **hard-coded numbers** ultimately traceable to `slow_kernels.csv` via the eval-section tables. Regenerating the figure means re-running the script after editing its hard-coded arrays. |

### RQs → evidence

| RQ | Where defined | Evidence base |
|---|---|---|
| RQ1 (Characterization) | `tex/RQ_summary.tex:20` | `ViperBench/` benchmark suite → `profile.csv` + `slow_kernels.csv` |
| RQ2 (Root cause) | `tex/RQ_summary.tex:22` | `experiments/` (ncu sweep, FP32 GEMM, Winograd isolation, autotune reconcile) |
| RQ3 (Mitigation) | `tex/RQ_summary.tex:24` | `AKO4ALL/results/` (5 campaigns) + `experiments/results/<gpu>/conv_mitigation*.csv` |

## Working in this repo

- Adding/modifying a kernel: keep the three backends' function name and signature identical; add `test_cases` covering small/medium/large/edge shapes; use `loose_tol=True` for reductions. Register tuning grids in `tuning/configs.py` if the kernel takes config params.
- TileLang gotcha worth remembering (it drove the biggest speedups here): `T.serial` reduction loops are slow — prefer `T.reduce`; use `T.Pipelined(num_stages=...)` to hide memory latency, `T.gemm` for matmul-heavy ops, and `T.use_swizzle` for L2 locality. Always `T.clear()` an accumulator before `T.gemm`. Deeper notes: `AKO4ALL/context/tilelang_reference.md` and `triton_tuning.md`.
- Editing the paper: respect the writing-style rules in `paper-latex-project/CLAUDE.md`. Before changing any RC0/RC3/RC4 wording, re-read the "Corrected root-cause labels" callout at the top of this file and `REVISION_TODO.md`. Numbers in tables are hand-typed from `slow_kernels.csv` and `AKO4ALL/results/optimization_results.csv` — keep them in sync if you re-run the benchmarks.
- Replaying the rebuttal suite on a new GPU: follow `experiments/A100_H100_RUNBOOK.md`. Results auto-namespace under `experiments/results/<gpu_slug>/`, so Ada outputs are safe.
