# Reviewer Weakness Analysis — ASE 2026 Paper #4134

**Paper:** *An Empirical Study of GPU Kernel Performance Gaps in Modern Domain-Specific Languages*
**Artifact:** this repository (`ViperBench/` benchmark suite + `AKO4ALL/` agentic optimizer)
**Prepared:** 2026-05-25 — rebuttal-preparation aid

## How this was produced

1. Read all three reviews (`reviews.txt`) and the full 11-page draft (`ase26-paper4134.pdf`).
2. Cataloged every substantive weakness and mapped each to the specific paper section/table it concerns.
3. Audited the **artifact ground truth** for each weakness via three parallel role-based investigations — a *Methodology & Reproducibility* auditor, a *Coverage & Scope* auditor, and a *Root-Cause & Mitigation Validity* auditor — each reading actual code, data files, and logs and quoting `path:line` evidence.
4. Synthesized below: for each weakness — who raised it, what the paper claims, what the artifact actually shows, a verdict, and a concrete rebuttal action.

> ⚠️ Caveat carried through this whole analysis: the **repo-root scaffolding** (`run_all.py`, `profile_all.py`, `test_harness.py`, …) is stale dead code targeting a nonexistent `newBench/`. All ground truth lives under `ViperBench/` and `AKO4ALL/`.

## Reviewer overview

| Review | Merit | Core thesis |
|--------|-------|-------------|
| 4134A | **2 — Weak Reject** | Root-cause taxonomy conflates compiler vs. authoring; uneven evidence (RC3/RC4 unvalidated); baseline asymmetry; limited conv coverage. |
| 4134B | **2 — Weak Reject** | Benchmark construction/selection unclear; evaluation incomplete (conv filters, per-kernel element-wise); some root causes lack experiments. |
| 4134C | **3 — Weak Accept** | External validity (22 kernels, 1 GPU, 2 configs/cell, forward-only); correctness validation under-described; baselines uneven across categories. |

Three independent reviewers converge on three themes: **(1) root-cause soundness/attribution, (2) coverage/external validity, (3) baseline fairness + correctness rigor.**

---

## Executive summary

Verdict legend: ✅ reviewer factually correct (artifact agrees) · ⚠️ partly correct (important nuance) · ❌ reviewer mistaken (artifact refutes).
Effort to address: 🟢 free (answer already in artifact) · 🟡 cheap (infra/kernels exist, just run/measure) · 🔴 substantial new work.

| ID | Weakness | Raised by | Verdict | Effort | Priority |
|----|----------|-----------|---------|--------|----------|
| **W1** | RC0 conflates compiler vs. kernel-authoring; FP32 failure not root-caused | A | ✅ | 🟢 provenance / 🔴 FP32 cause | **High** |
| **W2** | Baseline asymmetry; "library efficiency" means different things | A, C | ⚠️ | 🟢 LayerNorm point / 🔴 fused baselines | **High** |
| **W3** | RC2b/RC3/RC4 unvalidated; Nsight counters never shown | A, B | ✅ | 🔴 | **High** |
| **W4** | Manual rewrites recover perf, but no in-DSL/compiler fix | A | ✅ | 🟢 framing only | Med |
| **W5** | Limited scope: 1 GPU, 2 configs/cell, simple kernels, forward-only | A, B, C | ✅ | 🔴 (esp. 2nd GPU) | Med |
| **W6** | Conv claimed 1×1–7×7 + depthwise/strided; only 3×3 stride-1 benchmarked | A, B | ✅ | 🟡 | **High** |
| **W7** | "Heuristic tuning Δ=0pp" (§5) vs "1.66× matmul" (§7.3); tuning undefined | A, B | ✅ | 🟢 | **High** |
| **W8** | GPU clocks not locked; 9% run-to-run variance | A | ✅ | 🟡 | Med-High |
| **W9** | Benchmark construction / selection criteria / representativeness | B, C | ✅ | 🟡 authoring | Med |
| **W10** | No per-kernel results for the 15 element-wise kernels | B | ✅ | 🟢 | **High** |
| **W11** | Correctness validation under-described; FP32 failure; mitigations revalidated? | A, C | ⚠️ | 🟢 describe / 🔴 FP32+edges | **High** |
| **W12** | "Iteration" (LayerNorm 18, argmax 13) undefined | A | ✅ (of paper) | 🟢 | Med |
| **W13** | RC3 cites A100 registers though experiments are on Ada | A | ⚠️ benign | 🟢 | Low |

**Bottom line:** Most reviewer concerns are *factually correct against the artifact*. The good news for the authors: **several are free or cheap wins** (W7, W10, W12, W13, the LayerNorm half of W2, much of W6) because the data/answer already exists in the artifact or the kernels already support the missing configs. The genuinely hard ones (**W3, second-GPU part of W5, FP32 root-cause in W1/W11, fused baselines in W2**) require new experiments and are the ones most threatening to acceptance.

---

## Theme 1 — Root-cause taxonomy & evidence (W1, W3, W4, W13)

### W1 — RC0 conflates compiler-level vs. kernel-authoring issues; FP32 failure not root-caused
**Raised by:** 4134A (reason-to-reject #1 + Question 1).
**Paper says:** §6 titles RC0 "TileLang **Compiler-Level** Deficiencies," comprising (a) `T.serial` reduction loops instead of native `T.reduce`, and (b) absent 128-bit (LDG.128) loads. §6 also reports a separate FP32 GEMM correctness failure (99.6% mismatch, 2067× max rel. error at 4096×2048×1024), declared "orthogonal" and excluded. The paper itself admits the fix "does not represent a new optimization technique: `T.reduce` already existed in the API."

**Artifact shows:**
- The current ViperBench TileLang norm kernels **still use `T.serial`** — i.e. they *are* the unoptimized baseline: `ViperBench/layer_norm/tilelang_impl.py:44-45,50-51` (`for j in T.serial(n): mean_val[0] += row[j]`), `ViperBench/rms_norm/tilelang_impl.py:39-40`. The AKO4ALL-optimized versions swap in `T.reduce(...)`: `AKO4ALL/results/optimized/layer_norm_tilelang.py:32,47`, `rms_norm_tilelang.py:34`.
- The project's own guide states this is an **authoring** anti-pattern, not a compiler limit: `AKO4ALL/context/tilelang_reference.md:7` — "`T.reduce(...)` — parallel tile reduction (**NEVER use T.serial for reductions**)"; `:14` lists `T.serial` reductions as a *Common Performance Issue*. So `T.reduce` was always available; the 208× win came from changing the **kernel source**.
- **Genuine nuance that supports the reviewer's "conflation" charge:** the artifact *also* contains a real compiler-side complaint about `T.reduce`'s own butterfly lowering (`GPU Kernel Performance Analysis Report.md:73-81`; a filed upstream issue in `AKO4ALL/context/known_github_issues.md:1-69`) — but that is **not** what the mitigation fixed. RC0 thus bundles (i) an authoring fix that produced the 208×, with (ii) a real-but-unaddressed codegen issue.
- **FP32 failure:** appears in the artifact in exactly one place — a verbatim copied GitHub issue, `AKO4ALL/context/known_github_issues.md:71-160` ("Mismatched elements 4179519/4194304 (99.6%) … Greatest relative difference 2067.27 … float16 100% pass"). **No root cause is offered.** The nearest mechanistic hint is unconnected: `ViperBench/batched_matmul/tilelang_impl.py:10` — "We avoid T.gemm to keep full float32 precision (**T.gemm uses TF32**)." There is **no fp32 matmul test and no recorded fp32 failure** in any results file (`ViperBench/matmul/test.py:30-36` is fp16-only; all 44 result JSONs are PASS).

**Verdict:** ✅ Both halves of the critique hold. The label "Compiler-Level" overstates the compiler's role for the part that actually drove the headline number, and the FP32 issue is reported as a symptom with no attribution.

**Rebuttal action:**
- 🟢 *Disambiguate RC0 (no new experiments):* relabel/split it — (a) **kernel-authoring/example-code** issue (`T.serial` → `T.reduce`, the 208×, mechanically correctable in user space), vs. (b) **code-generation** issue (absent LDG.128 vectorization). Cite the artifact's own `tilelang_reference.md` and the upstream `T.reduce` issue to show you can tell them apart. This actually *strengthens* the paper.
- 🔴 *FP32 root cause (new work):* run a controlled experiment to localize it — e.g. compare `T.gemm` (TF32 path) vs. a non-`T.gemm` fp32 accumulation loop at 4096×2048×1024; the `batched_matmul` comment suggests TF32 truncation as the prime suspect. Add an fp32 matmul correctness case so the failure is reproducible from the artifact.

### W3 — RC2b/RC3/RC4 lack mitigation; Nsight counter data never shown
**Raised by:** 4134A, 4134B (+ 4134A Q4 asks for specific counters).
**Paper says:** §3.3 introduces Nsight Compute profiling across memory/compute/instruction counter classes; §3.4 says counters are "used qualitatively … not tabulated separately." Table 5 marks RC2b, RC3, RC4 with "—" (no direct mitigation).

**Artifact shows:**
- The 68 KB `AKO4ALL/context/GPU Kernel Performance Analysis Report.md` is **entirely qualitative** — **zero measured counters**. No `l1tex__*`, `sm__*`, `dram__*`, L2-hit-rate, warp-stall, or register-spill *measurements* anywhere. It cites hardware constants ("48 MB L2" `:17`, "64K 32-bit registers" `:24`, "360 GB/s" `:15`) and *derived* figures ("8·log(256/32)=24 thread syncs" `:79`), not profiler output. `ncu` appears only as methodology intent (`AKO4ALL/TASK.md:24`, `HINTS.md:3-4`).
- **RC3** (5×5+ register pressure): no register/spill/occupancy measurement; 5×5/7×7 conv is **never performance-benchmarked** (benchmark harnesses hardcode `3,3` — `ViperBench/benchmark.py:133,137`). A 5×5 case exists only as a toy-shape correctness test (`ViperBench/conv2d/test.py:34-39`).
- **RC4** (no Winograd): **no isolation experiment** (no cuDNN algorithm query, no Winograd-disable run) anywhere.
- `AKO4ALL/results/optimization_results.csv` has exactly **5** campaigns: layer_norm, rms_norm, argmax (RC0), conv2d, matmul (RC1/RC2a). RC2b, RC3, RC4 → genuinely unmitigated, matching Table 5's "—".

**Verdict:** ✅ Fully correct, and arguably the most acceptance-threatening cluster. The paper presents a counter-grounded root-cause taxonomy, but the artifact contains no counter tables, and three of the root causes rest on reasoning rather than data.

**Rebuttal action:** 🔴 Substantial new work. The infra exists (`ncu` workflow), so the authors *can* generate: (i) representative counter tables — vectorized-load fraction (conv vs. GEMM), register-spill (`sm__`), warp-stall breakdown for RC0 — exactly reviewer Q4's list; (ii) RC3 — actual 5×5/7×7 conv runs with register/occupancy counters; (iii) RC4 — a cuDNN run with Winograd disabled to isolate its contribution; (iv) RC2b — an L2-persistence experiment at 16384². None of this data exists today.

### W4 — Manual rewrites recover perf, but no demonstration the DSL/compiler can do it
**Raised by:** 4134A.
**Paper says:** §7 mitigations recover ≥95% efficiency for 4/5 kernels but are hand-written; §8.1 frames compiler/tooling fixes as future directions.
**Artifact shows:** All 5 mitigations in `AKO4ALL/results/optimized/` are **pure user-space DSL kernel rewrites** — `T.serial`→`T.reduce`, dtype changes, `T.rsqrt`, block-size retuning (TileLang); added `@triton.autotune` config lists + `GROUP_SIZE_M` swizzle + implicit-GEMM reformulation (Triton, `matmul_triton.py:10-26`, `conv2d_triton.py:10-32,128-139`). No file touches framework/compiler source (no `site-packages`, `.cu`, `ptxas`, monkeypatch). `AKO4ALL/HINTS.md:9` explicitly forbids leaving the DSL. The report's "compiler must be modified" passages (`:143-168`) are labeled theoretical/future.
**Verdict:** ✅ Correct — the reviewer's suspicion is exactly what the artifact shows.
**Rebuttal action:** 🟢 Framing, not new evidence. Position the rewrites honestly as **lower bounds on recoverable performance** and point to the filed upstream `T.reduce` issue (`known_github_issues.md`) as the systematic-fix path. Do not claim in-compiler fixes; that would require new work absent from the artifact.

### W13 — RC3 cites A100 register characteristics though experiments run on Ada (sm_89)
**Raised by:** 4134A.
**Paper says:** §6 RC3 — "Each streaming multiprocessor on **A100** has 65,536 32-bit registers per block."
**Artifact shows:** The artifact's own analysis states the spec **as an Ada figure and correctly**: `GPU Kernel Performance Analysis Report.md:24` ("The **Ada** SM provides 64K 32-bit registers … 48 concurrent warps/SM"), `:114` ("64K register limit of the **Ada** SM"). The per-SM register count (65,536) and per-thread cap (255) are **identical on A100 and Ada**; only max warps/SM differs (64 vs 48) and the report already uses the Ada value.
**Verdict:** ⚠️ The reviewer is right that the *paper* mislabels it "A100," but the characteristic transfers — it's a benign citation slip, not a logic error.
**Rebuttal action:** 🟢 One-line fix: relabel "A100" → "Ada" (or "A100/Ada — identical here") and cite your own report's Ada figures. *Note:* this does **not** repair W3's lack of measured register/spill data for RC3.

---

## Theme 2 — Baseline fairness & the "library efficiency" metric (W2)

### W2 — Baseline asymmetry across categories; aggregate "~65%" misleading
**Raised by:** 4134A, 4134C.
**Paper says:** §3.2 — GEMM = cuBLAS via `torch.matmul`; Conv = `nn.Conv2d` (NHWC, `allow_tf32=False`); Normalization = `F.layer_norm`/`F.rms_norm`; element-wise = "standard PyTorch eager." §3.5 — `cudnn.benchmark=False`. Table 4 aggregates "Overall (excl. attn) ~65% Triton / ~30% TileLang."

**Artifact shows:**
- Baselines per category (each `pytorch_impl.py`): GEMM `torch.matmul` (`matmul/pytorch_impl.py:6`) ✓ cuBLAS; Conv `F.conv2d` (`conv2d/pytorch_impl.py:14`) ✓ cuDNN; **LayerNorm `F.layer_norm` (`layer_norm/pytorch_impl.py:12`) — the FUSED library call, NOT unfused eager**; **RMSNorm hand-rolled eager** (`rms_norm/pytorch_impl.py:9-10`; `F.rms_norm` is used *nowhere* in the artifact); element-wise = eager `torch.add`/`F.relu`/`F.softmax`/etc.
- ⚠️ **Methodology claims not reflected in code:** a whole-repo grep finds **no `torch.backends.cudnn.benchmark`, no `allow_tf32`, no `set_float32_matmul_precision`** in any benchmark script — these are left at PyTorch defaults, not explicitly set as §3.2/§3.5 state. Conv inputs are fed **NCHW** (`benchmark.py:131-138`, `.contiguous()`), **not NHWC**. (The only `tf32=False` lives inside the Triton conv/attention kernels, not the cuDNN baseline.)

**Verdict:** ⚠️ Mixed, and it cuts both ways:
- The reviewers' blanket "normalization uses unfused eager" is **half-wrong** — LayerNorm uses fused `F.layer_norm`, so Triton's 94.6% there is a *fair* comparison. The authors have a clean correction here.
- But the deeper critique stands: RMSNorm and element-wise baselines *are* unfused eager (the paper even attributes RMSNorm's 1099% to lack of fusion), so a single "library efficiency" label spans cuBLAS, cuDNN, fused-library, and eager denominators — making "Overall ~65%" a blend of incomparable ratios.
- Separately, the **NHWC / `allow_tf32=False` / `cudnn.benchmark=False` statements are not reproducible from the artifact** — an accuracy problem independent of the reviewers.

**Rebuttal action:**
- 🟢 Correct the record: LayerNorm baseline is the **fused** `F.layer_norm` (rebut the "unfused" claim for it).
- 🟡/🔴 Adopt 4134C's suggestion: report **separate efficiency metrics** (vendor-library vs. fused-library vs. eager-PyTorch) instead of one blended aggregate; for element-wise/RMSNorm, add a fused baseline (e.g. `torch.compile`) to show the gap isn't a fusion artifact.
- 🔴 **Fix the methodology mismatch:** either actually set `cudnn.benchmark`, `allow_tf32=False`, and `channels_last`/NHWC in the benchmark and re-run, **or** correct §3.2/§3.5 to describe what the code does. (See also audit finding N1.)

---

## Theme 3 — Coverage & external validity (W5, W6, W9, W10)

### W5 — Limited experimental scope
**Raised by:** all three.
**Paper says:** §8.2/§10 already concede single GPU (RTX 4000 Ada), forward-pass only, ~2 configs/cell, dataset-size limits, and promise rebuttal-phase expansion.
**Artifact shows:** Exactly **2 configs per kernel** (`small`/`large`) for all 22 — uniformly 10 rows each in `profile.csv` (2 sizes × 5 impls). **15 of 22** kernels are simple element-wise/reduction/memory ops; only 7 are compute-heavy. **Forward-only confirmed:** benchmarked entry points are `cross_entropy_fwd`/`attention_fwd`; the few `backward()` defs (`rms_norm/triton_impl.py:43`, `log_softmax/triton_impl.py:70-96`) are never timed. Single arch only. `profile.csv` carries only `latency_ms` + `peak_memory_mb` (no throughput/efficiency column).
**Verdict:** ✅ Precisely correct (and partly already conceded by the paper).
**Rebuttal action:** 🔴 The honest move is to **calibrate claims** to the evaluated scope (the paper already does this in §10 — lean on it) and deliver the promised expansion. A **second GPU (A100/H100)** is the single highest-value addition — all three reviewers ask about generalization, and §10 explicitly flags it. That data does not exist in the artifact yet.

### W6 — Convolution coverage: 1×1–7×7 + depthwise/strided claimed, only 3×3 stride-1 benchmarked
**Raised by:** 4134A, 4134B (+ both Q2).
**Paper says:** §3.1 claims conv covers "1×1 to 7×7 filters, including depthwise and strided cases"; Table 2 reports only two configs, both **3×3 stride-1**.
**Artifact shows:**
- Benchmarked conv = **only 3×3 stride-1 pad-1**, two shapes: `(8,64,56,56)` and `(32,256,128,128)` (`profile.csv:42-51`; harnesses hardcode `(.,.,3,3)` + `padding=1`). AKO4ALL conv mitigation also only 3×3 (`prepare_kernel.py:52-61`).
- **Partial mitigation:** the kernels **support arbitrary filter/stride/groups** (`conv2d/{triton,tilelang}_impl.py` have `groups` code paths; dilation explicitly unsupported), and **correctness tests already exercise 5×5 and stride-2** at toy shapes — `conv2d/test.py:14-50` (`5x5_kernel`, `stride2_pad1`), logged passing in `conv2d.json`/`conv2d_tilelang.json` (5×5 max_err 0.0126).
- **1×1, 7×7, and depthwise (groups>1) are entirely absent** — not even a correctness test.
**Verdict:** ✅ Correct for the *performance* claims; partly softened because implementations support the range and 5×5/stride-2 were correctness-validated (at toy scale).
**Rebuttal action:** 🟡 **Cheap and high-value** — the kernels already support these filters, so producing benchmark numbers for 1×1, 5×5, 7×7 (and a depthwise case) at real shapes is a config change to `benchmark.py`, not new kernel work. This directly answers A-Q2 and B-Q2 and is one of the best effort-to-impact fixes available.

### W9 — Benchmark construction / selection criteria / representativeness
**Raised by:** 4134B (reason-to-reject + Q1), 4134C.
**Paper says:** §3.1 — derived from TritonBench, the TileLang example repo, and own implementations; excludes sparse/runtime-dependent shapes.
**Artifact shows:** **No provenance anywhere** in the ViperBench kernels — grepping for `TritonBench`/"derived from"/source URLs across `ViperBench/**` yields zero attributions; kernel headers are uniform boilerplate. One kernel silently reveals its origin (`layer_norm/triton_impl.py:4-9` imports `torch._inductor` → a `torch.compile`-generated kernel) but is uncommented. No manifest/selection doc exists. (All TritonBench/KernelBench references belong to `AKO4ALL/`, the optimizer, not the suite's selection.)
**Verdict:** ✅ Correct — selection criteria, source mapping, and representativeness justification are genuinely undocumented.
**Rebuttal action:** 🟡 **Authoring, not experiments** — write the provenance table (which kernel came from which source), the selection criteria, and the representativeness argument (operator-category coverage). This must come from the authors' records; it can't be reconstructed from the artifact. Documenting that `layer_norm` Triton is Inductor-generated is a good concrete start.

### W10 — No per-kernel results for the 15 element-wise kernels
**Raised by:** 4134B (Q3).
**Paper says:** reports only category-level element-wise results.
**Artifact shows:** **The per-kernel data already exists** in `profile.csv` — all 15 kernels, 10 rows each, plus a digested `slow_kernels.csv`. Large-shape latencies (PyTorch / Triton-tuned / TileLang-tuned, ms) are directly tabulatable, e.g. `add` 1.33/1.28/1.71, `leaky_relu` 72.0/30.0/19.7, `softmax` 1.75/1.79/8.70, `argmax` 1.62/2.26/25.1, `cross_entropy` 23.86/1.87/27.77 (✓ the old 15908 ms PyTorch figure was the unvectorized-loop reference — a contamination artifact; the reference is now vectorized and `profile.csv` is **patched** → 23.8647 ms, E_lib sane at 1277%/86%), `embedding` 6.93/1.71/12.0 (full set in the Coverage audit / `profile.csv:2-221`).
**Verdict:** ✅ Correct, and a **free win**.
**Rebuttal action:** 🟢 Drop a per-kernel table straight from `profile.csv`/`slow_kernels.csv`. *Only* gap: if an efficiency/throughput metric (not just latency) is wanted, that column doesn't exist and must be derived from byte/FLOP counts.

---

## Theme 4 — Reproducibility & measurement (W7, W8, W11)

### W7 — "Heuristic tuning Δ=0pp" (§5) vs "1.66× matmul" (§7.3); "heuristic tuning" undefined
**Raised by:** 4134A, 4134B (+ both Q on tuning).
**Paper says:** §5 — all 22 kernels re-evaluated with "heuristically tuned configurations," Δ=0pp. §7.3 — `@triton.autotune` (12 configs) + `GROUP_SIZE_M` L2 swizzle gives matmul 2.71→1.63 ms (1.66×) on **4096×4096**. The RQ1 31.7% gap is on **16384²**.
**Artifact shows (the contradiction is fully explainable):**
- "Heuristic tuning" = the `ViperBench/tuning/` grid sweep caching **one** best config per kernel. The matmul grid is **12 block-tile configs only** — `bm/bn/bk` over {32,64,128}², `[:12]`, **no `GROUP_SIZE_M`, no `num_warps`, no `num_stages`, no swizzle** (`tuning/configs.py:19-23`). It is **swept on 4096²** (`tuning/sweep.py:69-70`) but **applied to the 16384² benchmark**.
- Result on 16384²: `profile.csv` matmul-large `triton`=361.81 vs `triton_tuned`=361.90 ms (Δ≈0) — corroborates §5.
- The richer §7.3 mechanism **does exist in the artifact**, just not wired into the main benchmark: `AKO4ALL/results/optimized/matmul_triton.py:10-26` carries the 12-config `@triton.autotune` + `GROUP_SIZE_M`. The plain `ViperBench/matmul/triton_impl.py` has no autotune. So §5 and §7.3 use **different search spaces on different shapes** — not a genuine inconsistency.
**Verdict:** ✅ The reviewers are right that "heuristic tuning" is undefined and the two numbers look contradictory — but the artifact resolves it cleanly.
**Rebuttal action:** 🟢 Define "heuristic tuning" precisely (quote `configs.py:19-23`: 12 block-tile configs, chosen `64×128×32` per `tuning_cache.json:113-117`), show via `profile.csv` it yields Δ≈0 on 16384², and explain §7.3 is a *larger* search (adds `GROUP_SIZE_M`/swizzle) on the *smaller* 4096² shape (code at `AKO4ALL/results/optimized/matmul_triton.py`). Also address *why* the expanded search wasn't applied in RQ1 (it was scoped to mitigation). Note the cached config was selected on 4096² but reported as Δ=0pp on 16384² — consider re-sweeping at the benchmark shape.

### W8 — GPU clocks not locked (9% run-to-run variance)
**Raised by:** 4134A.
**Paper says:** §3.3 "verify stability within 2% across five repeated runs"; Table 7 footnote attributes a 9% conv difference to "run-to-run GPU clock variation." No mention of locking clocks.
**Artifact shows:** **No clock locking, persistence mode, or `nvidia-smi` of any kind** anywhere (whole-repo grep for `-lgc`/`-pm`/`setGpuLockedClocks`/application clocks → nothing). Timing is `torch.cuda.synchronize()` + **`time.perf_counter()`** (host wall-clock, *not* CUDA events), median of **100** iters after 10 warmup (`benchmark.py:20-58`); slow kernels use 5 iters / 3 warmup. No thermal control, no cooldown, no recorded clock/variance telemetry; the "2%" and "9%" figures are not logged or reproduced in the artifact.
**Verdict:** ✅ Fully correct — under unlocked clocks, the 9% session variance is plausible and small differences (94.6% vs 97.8%) may be within noise.
**Rebuttal action:** 🟡 New but cheap: lock clocks (`nvidia-smi -pm 1` + `-lgc <freq>`) and re-measure key comparisons, and/or report std-dev / confidence intervals across repeated full runs. Optionally switch to CUDA-event timing for kernel isolation. As-is the artifact cannot show small differences are significant.

### W11 — Correctness validation under-described; FP32 failure; mitigations revalidated?
**Raised by:** 4134C (main concern + Q), 4134A (FP32).
**Paper says:** §6 reports the FP32 GEMM failure and excludes it; correctness methodology is otherwise thin.
**Artifact shows:**
- Tolerance table (`test_utils.py:12-25`): fp32 atol/rtol 1e-5, fp16 1e-3, bf16 1e-2; `loose_tol` doubles them. Comparison = `torch.allclose(ref.float(), test.float())` + max-abs-err.
- ⚠️ **Non-uniform tolerances** beyond the headline table: matmul uses `atol=0.2, rtol=1e-2` (`matmul/test.py:20`), conv `atol=2e-2` (`conv2d/test.py:58`) — should be disclosed.
- Inputs are **almost entirely `torch.randn`** (98 uses); only **3 hand-crafted edge tensors** (relu sign patterns). No NaN/Inf/large-magnitude/denormal stress tests. 4–9 shapes per kernel.
- **All 44 result JSONs report PASS; zero recorded failures.** The **FP32 GEMM failure is absent from the artifact** — no fp32 matmul test (matmul suite is fp16-only), no logged 99.6%/2067× result (those strings live only in `known_github_issues.md` + `reviews.txt`).
- **Mitigation kernels:** the AKO4ALL-optimized kernels are *not* re-run through ViperBench's `test_utils` harness. However, they **were** correctness-checked by AKO4ALL's KernelBench evaluator during optimization — the iteration logs record INCORRECT/FAILED attempts (e.g. argmax iters 1, 11, 12), and `TASK.md` mandates `CORRECT=True` at baseline.
**Verdict:** ⚠️ Reviewer's "under-described" concern is correct; the artifact answers *some* of it (tolerances, comparison method, distributions, shape counts) but confirms the gaps (Gaussian-only inputs, the FP32 failure not reproducible here, mitigations not revalidated *in ViperBench*).
**Rebuttal action:**
- 🟢 Describe the validation design from the artifact (tolerance table + per-kernel overrides + shape counts) and note mitigation kernels were validated by AKO4ALL's evaluator (cite the correctness rows in the iteration logs).
- 🔴 New work: add an fp32 matmul test so the failure is reproducible and root-caused (ties to W1); add edge-case inputs (NaN/Inf/large/denormal); and ideally re-run the 5 mitigation kernels through ViperBench's harness for a uniform correctness record. Standardize/disclose the per-kernel tolerance exceptions.

---

## Theme 5 — Mitigation clarity (W12)

### W12 — "Iteration" (LayerNorm 18, argmax 13) undefined
**Raised by:** 4134A.
**Paper says:** mentions "18" and "13" iterations without defining one.
**Artifact shows:** A **precise definition** in `AKO4ALL/TASK.md:28-37`: "Every modification to `solution/` code followed by a `bash scripts/bench.sh` run counts as one iteration — regardless of improvement, regression, or failure," with three required steps (benchmark `iter-N` → update `ITERATIONS.md` → git commit). Corroborated by `AKO4ALL/README.md:64-65`. The counts **reconcile exactly**: `layer_norm_iterations.md` lists iters 1–18; `argmax_iterations.md` 1–13 (incl. failed attempts); matmul 6, rms_norm 2, conv2d 15 — all matching `optimization_results.csv`.
**Verdict:** ✅ The *paper* is under-specified, but the artifact removes all ambiguity.
**Rebuttal action:** 🟢 Add one sentence citing `TASK.md`: *"One iteration = one manual kernel-source edit + one benchmark run (with correctness check), logged and committed; failed/regressing attempts count; autotuning sweeps occur within a single iteration."* Reproduce the iteration tables from `*_iterations.md`.

---

## Additional discrepancies surfaced by the audit (not raised by reviewers)

These are artifact-vs-paper mismatches the reviewers did **not** catch but that the authors should fix proactively — some would be damaging if a reviewer found them during rebuttal.

- **N1 — Methodology statements unreproducible (most important).** §3.2/§3.5 claim `cudnn.benchmark=False`, `allow_tf32=False`, and **NHWC** conv inputs. The benchmark code sets **none** of these and feeds **NCHW** (see W2). Either set them and re-run, or correct the text. This directly weakens any W2 rebuttal if left unaddressed.
- **N2 — §7.3 matmul variant lives in `AKO4ALL/`, not the benchmark.** The `@triton.autotune`+`GROUP_SIZE_M` matmul (2.71→1.63 ms) is at `AKO4ALL/results/optimized/matmul_triton.py`, while `ViperBench/matmul/triton_impl.py` is the plain kernel. The §7.3 result *is* reproducible, but make the provenance explicit so the §5/§7.3 numbers are traceable.
- **N3 — Input-shape docs disagree with measured shapes.** `kernel_input_shapes.html` lists attention-large `(32,32,4096,128)` and batched_matmul-small `A:(64,128,128)`, but `profile.csv` actually measured `(8,32,2048,128)` and `A:(64,128) B:(64,128,128)`; `prepare_kernel.py` uses yet another attention shape `(8,32,2048,64)`. If Table 1 cites the HTML shapes, it will mismatch the results. Reconcile to one source of truth.
- **N4 — No efficiency/throughput column.** `profile.csv` stores only latency + peak memory. `E_lib` (a latency ratio) is fine, but any hardware-efficiency/bandwidth/roofline table (reviewer Q4) must be derived or freshly measured.
- **N5 — Mitigation kernels not in the ViperBench correctness record.** They were validated by AKO4ALL's evaluator, but a uniform re-run through `ViperBench/test_utils.py` would let you answer reviewer C's "were mitigation kernels revalidated?" cleanly.

---

## Minor issues

- **M1 — Kernel count 21 vs 22.** Artifact has **22** kernel dirs with `test.py` (GEMM 3: `matmul`, `batched_matmul`, `linear_activation`; attention 1; conv 1; norm 2; element-wise/reduction 15). The intro's "21" is a typo → make it 22 everywhere.
- **M2 — Typo "anamoly" → "anomaly"** (abstract).
- **M3 — Table 1 notation.** "16384²" = a 16384×16384 square fp16 matmul (`profile.csv:137-141`); "64×128²" = batched matmul, batch 64 of 128×128 (`A:(64,128) B:(64,128,128)`, `profile.csv:32-36`); the large batched case is "128×2048²". Spell these out in the caption.

---

## Prioritized rebuttal action plan

**Free wins — answerable now from the artifact (do all of these):**
- W10: add the per-kernel element-wise table (data in `profile.csv`). 
- W12: define "iteration" from `TASK.md`; reproduce iteration tables.
- W7: define "heuristic tuning" (12-config grid) and explain the §5/§7.3 shape+search difference.
- W13: relabel A100→Ada.
- W2 (partial): correct the "LayerNorm uses unfused eager" misconception.
- W1 (partial): split RC0 into authoring vs. codegen.
- M1/M2/M3: fix count, typo, notation.

**Cheap experiments — infra/kernels already exist (high effort-to-impact):**
- W6: benchmark 1×1 / 5×5 / 7×7 / depthwise conv at real shapes (kernels already support them).
- W8: re-measure key comparisons with locked clocks + error bars.
- W9: write the provenance + selection-criteria documentation.
- N1: set the cuDNN/tf32/NHWC flags and re-run conv (or correct the text).

**Substantial new work — most acceptance-critical, plan for the rebuttal window:**
- W3: collect and tabulate Nsight counters (vectorized-load fraction, register spill, warp stalls); run RC3 (5×5/7×7 register/occupancy), RC4 (cuDNN Winograd-disable), RC2b (L2 persistence at 16384²).
- W5: add a second GPU (A100/H100) — every reviewer asks about generalization.
- W2 (full): report split metrics (vendor / fused-library / eager) and add fused baselines for element-wise + RMSNorm.
- W1/W11 (FP32): reproduce and root-cause the FP32 GEMM failure (likely `T.gemm` TF32 path); add fp32 + edge-case correctness tests; revalidate mitigation kernels.

*Strategic note:* the three Weak-Reject-driving themes are **root-cause soundness (W1, W3, W4)**, **baseline fairness (W2)**, and **coverage/correctness (W5, W6, W11)**. W3 and the FP32 root-cause are the highest-risk items because the paper's central contribution is a *counter-grounded* root-cause taxonomy, yet the artifact contains no measured counters and no FP32 attribution. Prioritizing measurable counter data (W3) and the cheap coverage win (W6) likely moves the needle most.
