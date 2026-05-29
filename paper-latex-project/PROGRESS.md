# PROGRESS.md — ASE '26 Paper Progress Tracker

**Title:** An Empirical Study of GPU Kernel Performance Gaps in Modern Domain-Specific Languages
**Venue:** ASE 2026 (sigconf, anonymous review mode)
**Date last updated:** 2026-03-26

---

## Overall Status

| Phase | Status |
| --- | --- |
| Paper structure & scaffolding | DONE |
| Writing (prose, no data) | DONE |
| Experimental data collection | IN PROGRESS |
| Figure/table population | IN PROGRESS |
| Bibliography cleanup | IN PROGRESS |
| Final polish & submission | NOT STARTED |

---

## Section-by-Section Status

| File | Status | Notes |
| --- | --- | --- |
| `tex/abstract.tex` | COMPLETE | Numbers inserted; recovery % still pending experiments |
| `tex/introduction.tex` | COMPLETE | RQs, contributions, paper outline all present |
| `tex/background.tex` | COMPLETE | Covers tile model, Triton, TileLang, cuBLAS/cuDNN, TritonBench |
| `tex/methodology.tex` | DRAFT — needs version strings | TODOs: PyTorch/CUDA/Triton/TileLang/Nsight versions; locked GPU clock freq; repo URL |
| `tex/evaluation.tex` | DRAFT — data populated | Tables filled from profile.csv; H100 and element-wise detail table pending |
| `tex/analysis.tex` | DRAFT — RC0 added | RC0 new finding, RC1–RC4 marked [counter pending] |
| `tex/mitigation.tex` | SCAFFOLD | 3 mitigations written (M1–M3); all recovery numbers pending data |
| `tex/discussion.tex` | DRAFT — mostly complete | One TODO: H100 vs. A100 TMA discussion needs verification |
| `tex/related_work.tex` | COMPLETE | Covers DSL landscape, conv optimization, profiling tools, benchmarking |
| `tex/threats.tex` | COMPLETE | 4 validity threats addressed |
| `tex/conclusion.tex` | DRAFT — numbers inserted | 21 kernels, conv efficiency numbers added; recovery % still pending |

---

## Open TODOs (by priority)

### Blocking (need experimental data)

- [ ] Run benchmark: collect TFLOPS for GEMM, Attention, Conv2d, Normalization, Element-wise (Triton + TileLang vs. cuBLAS/cuDNN) on A100 and H100
- [ ] Run Nsight Compute profiles: collect vectorized-load fraction, cp.async count, warp stall cycles, register spill, SM occupancy per kernel category
- [ ] Populate `tab:rootcauses` — contribution % for RC1–RC4
- [ ] Populate `tab:mitigation` — before/after mitigation table
- [ ] Generate `figures/overview_efficiency.pdf` — violin/box plot
- [ ] Run H100 experiments for A100 vs. H100 microarchitecture comparison (sec:eval:arch)
- [ ] Add element-wise detail table to evaluation.tex (currently only in tab:summary)
- [ ] Verify TileLang JIT overhead vs. warm-cache latency (RC0)

### Near-term (writing)

- [ ] Fill in software version strings in `tex/methodology.tex` (PyTorch, CUDA, Triton, TileLang, Nsight Compute, Python)
- [ ] Fill in hardware details (GPU models confirmed, driver version, host CPU)
- [ ] Fill in locked GPU clock frequency
- [ ] Insert repo URL (when created)
- [x] Fill kernel count N in abstract and conclusion
- [x] Insert headline percentages in abstract (DSL X% of cuBLAS on GEMM, Y% of cuDNN on conv)
- [ ] Insert recovery fraction in conclusion
- [x] Populate `tab:gemm`
- [x] Populate `tab:conv`
- [x] Populate `tab:summary`
- [x] Add normalization table (tab:norm) — added to evaluation.tex

### Verification needed (claims flagged in text)

- [ ] RC1: verify absent vectorization via Nsight Compute vectorized-load-fraction counter
- [ ] RC2: verify by sweeping broader auto-tune search space
- [ ] RC3: verify register spill and SM occupancy via Nsight Compute + cuobjdump
- [ ] RC4: estimate theoretical Winograd impact per configuration
- [ ] Eval finding: confirm $1\times1$ conv gap is narrower than $3\times3$ gap
- [ ] Eval finding: confirm H100 gap is smaller than A100 gap (TMA hypothesis)
- [ ] Mitigation M1: verify LDG.128 emission via cuobjdump after loop reorder + padding
- [ ] Mitigation M3: verify Winograd breakeven point across batch sizes
- [ ] Reconcile attention baseline in methodology.tex: claims `enable_flash_sdp(True)` but profiling used naive PyTorch path — update methodology.tex to reflect actual baseline

---

## Key Design Decisions (locked)

- **Kernel categories:** GEMM, Attention, Convolution, Normalization, Element-wise (5 categories)
- **DSLs:** Triton + TileLang
- **Baselines:** cuBLAS (GEMM), cuDNN (Conv/Attention), PyTorch eager (Element-wise)
- **Hardware:** NVIDIA A100-SXM4-80GB and H100-SXM5-80GB
- **Profiling:** CUDA events (throughput) + Nsight Compute (counters)
- **Primary metric:** Library efficiency % = DSL TFLOPS / lib TFLOPS × 100
- **Root causes:** RC1 absent vectorization, RC2 autotune mismatch, RC3 register pressure, RC4 no Winograd
- **Mitigations:** M1 vectorization-aware access, M2 extended search space, M3 Winograd lowering

---

## Build

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Current build: compiles with placeholder content (no figures directory needed — fallback boxes render inline).
