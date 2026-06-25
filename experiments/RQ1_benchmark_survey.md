# RQ1 Benchmark Survey — KernelBench & TritonBench (DSL kernel-quality evaluation)

> **Purpose (RQ1).** Characterize, precisely and fairly, what the two most-prevalent DSL / LLM-generated-kernel
> benchmarks — **KernelBench** (Stanford Scaling Intelligence) and **TritonBench** — actually evaluate, and
> where they fall short for judging whether a DSL kernel is *well-written* (i.e., near the vendor library) as
> opposed to merely *correct*. Per `PIVOT_FRAMING.md` Guardrail 1, the critique must be accurate, not a
> strawman: each subsection states what the benchmark **does well** before what it misses, and every claim is
> sourced (URL / paper / repo file:line). Unverifiable claims are tagged `[UNVERIFIED]`.

---

## 0. Method & scope

- Read the **vendored** KernelBench evaluator shipped in this repo (`AKO4ALL/bench/kernelbench/bench.py`,
  whose header states it "Inlines core logic from KernelBench's eval.py and timing.py") and confirmed each
  behavior against the **live upstream source** (`ScalingIntelligence/KernelBench`, branch `main`,
  `src/kernelbench/{eval.py,score.py}` and the adversarial unit tests).
- Read the **TritonBench** paper (arXiv:2502.14752, ACL Findings 2025) and noted the *separate* Meta
  performance suite of the same name (`meta-pytorch/tritonbench`) so the two are not conflated.
- "Does a slow-but-correct kernel pass?" is the load-bearing question, so the gating logic is quoted at
  file:line.

**Terminology caution (fairness).** "TritonBench" denotes **two distinct artifacts**: (a) the academic
LLM-Triton-generation benchmark of Hong et al. (arXiv:2502.14752), and (b) Meta's `tritonbench` operator
**performance** harness. This survey treats (a) as the benchmark parallel to KernelBench, and flags (b) where
relevant. Conflating them would be a strawman.

---

## 1. KernelBench (Stanford Scaling Intelligence)

### 1.1 What it is and what it does well

KernelBench is the de-facto standard benchmark for **LLM GPU-kernel generation**: 250 PyTorch ML workloads
across four levels — Level 1 (100 single-kernel ops), Level 2 (100 fusion patterns), Level 3 (50 full model
architectures), Level 4 (HuggingFace models). The model is given a reference `class Model(nn.Module)` and must
emit a `ModelNew` that reproduces its output faster. Strengths relevant to *performance* evaluation:

- **It does score performance, not just correctness.** Its headline metric `fast_p` is the *fraction of tasks
  that are correct **and** achieve speedup > `p` over the baseline*, where speedup = baseline_time / generated_time
  (`src/kernelbench/score.py:28–36`: `speed_up = filtered_baseline_speed / filtered_actual_speed;
  fast_p_score = np.sum(speed_up > p)`). So `fast_1` = correct **and** faster than the baseline, `fast_2` =
  correct **and** ≥2× — a real performance bar, not pure pass/fail.
- **A credible baseline, including `torch.compile`.** The baseline is **PyTorch eager**, and the repo also
  ships `torch.compile` (TorchInductor) baseline timings alongside eager (committed
  `results/timing/<gpu>/baseline_time_torch.json` **and**
  `baseline_time_torch_compile_inductor_default.json` for H100_Modal, H100_PCIe_LambdaLabs, etc.). So a
  generated kernel can be required to beat a *compiled* PyTorch reference, which is a stronger bar than eager.
- **Multi-trial correctness with documented anti-cheat.** Correctness runs `num_correct_trials` random-input
  trials and passes only if **all** trials match (`eval.py`: `if pass_count == num_correct_trials: ...correct`);
  it checks output-shape equality first, then `torch.allclose(output, output_new, atol=tolerance, rtol=tolerance)`.
  The repo carries an **adversarial unit-test suite** (`src/kernelbench/unit_tests/test_eval_adversarial.py`)
  that asserts the evaluator rejects three named reward-hacks: `test_result_cache_reuse` (uninitialized-output
  memory that happens to alias a prior result), `test_input_modification` (zeroing inputs so ref and candidate
  both return zeros), and `test_non_default_stream` (timing the kernel on a non-default stream to fake speedup).
- **Backend coverage has broadened beyond CUDA.** The current `main` branch advertises backends
  `cuda, triton, cute, tilelang, thunderkittens` (mirrored by the vendored copy's `--backend` choices
  `cuda, triton, tilelang, cute, hip`). So it is **not** accurate to call KernelBench "CUDA-only" today; that
  was the December-2024 launch state, and DSL/AMD expansion is the Fall-2025 roadmap (Issue #74).
- **Per-dtype tolerances tied to torchbench.** `get_tolerance_for_precision` (upstream `eval.py:93–100`):
  fp32 `1e-4`, fp16 `1e-2`, bf16 `1e-2`.

### 1.2 What it misses for *DSL performance-quality* evaluation

- **Correctness is the only hard gate; performance is a reported metric, not a pass threshold.** A correct
  kernel that is 100× slower than the vendor library is still a **valid, accepted, "correct" submission**: it
  is counted in `fast_0` (= correctness rate) and simply omitted from `fast_1`/`fast_2`. The benchmark never
  *rejects* a slow kernel — it just scores it low on a continuous axis. The vendored evaluator makes this
  concrete: its process exit code is `sys.exit(0 if result.correctness else 1)`
  (`AKO4ALL/bench/kernelbench/bench.py:1009`) — **purely correctness**, independent of speedup. A developer or
  CI using KernelBench-style "did it pass?" gating gets a green check for an arbitrarily slow kernel.
- **One hardcoded shape per task; no shape sweep.** Each problem fixes a single shape in module-level constants
  (e.g. Level-1 matmul: `N = 2048 * 2 = 4096`, `get_inputs()` returns two `torch.rand(N, N)`), so a kernel can
  be tuned to exactly one size and never tested for per-shape collapse. *(A secondary write-up claimed "3–5
  canonical shapes, geomean over shapes"; this is **contradicted** by the actual problem files, which define a
  single shape — so we do not rely on it.)*
- **fp32 by default.** `torch.rand` produces fp32 and reported results are fp32 ("Currently all of our reported
  results are fp32"); fp16/bf16 are supported but not the headline regime — so the dtype where DSLs most often
  win or lose (low-precision tensor-core paths) is under-exercised by default.
- **Speedup is relative to PyTorch on the *same* shape/GPU — it is **not** a vendor-library or roofline anchor.**
  `fast_p` answers "faster than this PyTorch run?", not "close to hardware peak / cuBLAS / cuDNN?". A kernel can
  clear `fast_1` while leaving most of the roofline on the table.
- **Anti-cheat is partial and the speedup guard only warns.** The excessive-speedup check
  (`check_for_excessive_speedup=True`, `excessive_speedup_threshold=10`) is annotated in upstream as
  *"Guard against potential reward hacking [optional but ongoing enhancement]"* and *"experimental"*; on a >10×
  result it only sets `metadata["excessive_speedup"]=True` and prints a `[WARNING]` (upstream `eval.py:691–692`;
  vendored `bench.py:725–734`) — **it does not fail the kernel.** This is exactly the loophole the Sakana "AI
  CUDA Engineer" episode exploited (memory-reuse / dropped-computation kernels reported as 10–150× "speedups"
  that were eval bugs, later walked back). KernelBench's own adversarial suite plugs specific known hacks but is
  explicitly an "ongoing enhancement," not a closed set.
- **Single-GPU-per-run, generation-target framing.** Cross-architecture portability of a kernel's *quality* is
  not part of the score (you re-run per GPU against that GPU's baseline file).

> **Vendored-copy caveat (verifiable internal inconsistency).** The repo's vendored evaluator is **looser** than
> upstream on correctness: `bench.py:259–265` returns fp32 `1e-3`, fp16 `5e-2`, bf16 `5e-2`, whereas its own
> `GUIDE.md:67–75` table advertises fp32 `1e-4`, fp16/bf16 `1e-2` (matching upstream). The code, not the doc,
> is what runs — so the vendored gate is even more permissive than the upstream benchmark it derives from.

---

## 2. TritonBench (Hong et al., arXiv:2502.14752)

### 2.1 What it is and what it does well

TritonBench is the first benchmark dedicated to **LLM generation of Triton operators**, with two channels:

- **TritonBench-G** — **184** real-world Triton operators curated from GitHub repos (>100 stars), filtered for
  `@triton.jit`, graded into five difficulty tiers d1–d5 by memory/scheduling complexity (attention, matmul,
  softmax, normalization, fused/pipelined kernels).
- **TritonBench-T** — **166** tasks aligned to PyTorch interfaces, synthesized by fusing high- and
  low-frequency PyTorch operators to cover ops under-represented on GitHub.

Strengths relevant to *performance* evaluation:

- **It explicitly measures performance, with a hardware-anchored efficiency metric.** Beyond Call Accuracy and
  Execution Accuracy (does it run / produce correct output), it reports **Speedup** (t_ref / t_gen) **and GPU
  Efficiency = measured performance ÷ the A100's theoretical peak**. That efficiency metric is effectively a
  **roofline anchor** — a *baseline-independent* notion of "well-written" — which is more than KernelBench's
  PyTorch-relative `fast_p` offers. (This is precisely the non-circular signal `PIVOT_FRAMING.md` argues for in
  RQ3, so TritonBench is partial corroboration, not just a foil.)
- **Realistic, difficulty-graded operator distribution** drawn from production Triton code, plus per-operator
  multi-branch correctness tests (avg ≈3.6 test branches/operator) to exercise code paths.
- **Honest about its own limits**: the paper states evaluation is on a **single NVIDIA A100** and flags hardware
  generalization as a limitation.

### 2.2 What it misses for *DSL performance-quality* evaluation

- **Triton only.** By construction it says nothing about TileLang, CUTLASS/CuTe, ThunderKittens, or any other
  DSL — so it cannot adjudicate cross-DSL kernel quality.
- **Single hardware (one A100).** No cross-architecture coverage; a kernel "efficient" on A100 may be far from
  peak on H100/Ada/GH200, which the benchmark cannot see.
- **No performance pass/fail gate.** Performance (speedup, efficiency) is **reported alongside** correctness;
  operators only need to execute correctly to be *evaluated* for speed — a slow-but-correct Triton kernel is
  still a successful generation on the correctness axis. Performance is a leaderboard number, not an admission
  bar.
- **Speedup is reference-relative, and the reference is the curated/PyTorch op, not necessarily the best vendor
  library.** TritonBench-G speedup is gen-vs-the-GitHub-reference-Triton-kernel; TritonBench-T is
  gen-vs-PyTorch. Neither pins the candidate to a tuned vendor library (cuBLAS/cuDNN/CUTLASS) as the bar.
- **Dtype coverage is not specified in the paper text we retrieved** — `[UNVERIFIED]`: the paper section
  fetched does not enumerate fp32/fp16/bf16 coverage; operator tests inherit each operator's native dtype, so
  dtype breadth is per-operator and not a controlled axis.
- **Shape coverage is per-operator correctness branches, not a controlled performance shape-sweep** — the
  "≈3.6 branches/operator" figure is about exercising code paths for *correctness*, not systematically
  measuring per-shape performance collapse.

---

## 3. Comparison table

Rows = the two surveyed benchmarks + **ViperBench** (this paper's *diagnostic probe*, included for contrast —
**not** proposed as a comprehensive benchmark; see `PIVOT_FRAMING.md` Guardrail 3). "Performance gate" answers
the load-bearing RQ1 question: *does a functionally-correct but slow kernel pass?*

| Dimension | **KernelBench** | **TritonBench** (acad., 2502.14752) | **ViperBench** (our probe) |
|---|---|---|---|
| **DSL coverage** | CUDA + PyTorch-ext at launch; `main` now adds **Triton, CuTe, TileLang, ThunderKittens** (Issue #74) | **Triton only** | PyTorch (ref) + **Triton + TileLang**, 22 kernels × 3 backends |
| **Dtype coverage** | fp32 default (reported results fp32); fp16/bf16 supported | `[UNVERIFIED]` — not enumerated; per-operator native dtype | fp32 / fp16 / bf16 (per-kernel, e.g. matmul fp16, layer_norm bf16) |
| **Shape coverage** | **1 hardcoded shape per task** (e.g. matmul `N=4096`) | per-operator correctness branches (avg ≈3.6), single A100 sizing | multiple shapes per kernel (small→large + 3-D/edge; matmul 64→4096) |
| **Hardware coverage** | 1 GPU per run vs that GPU's baseline file (H100/etc. baselines shipped) | **single A100** (stated limitation) | 4–5 GPU classes namespaced: RTX4000Ada, A100-PCIe, A100-SXM4, H100 (+GH200) |
| **Baseline** | **PyTorch eager + `torch.compile` (Inductor)** committed baselines | reference op (curated Triton for -G; PyTorch for -T) | PyTorch eager / cuDNN built-ins (`torch.compile` arm in `experiments/`) |
| **Performance gate (does slow-but-correct pass?)** | **YES, it passes.** Correctness is the only hard gate; `fast_0`=correctness; speed is a continuous `fast_p`; vendored exit code = correctness only (`bench.py:1009`) | **YES, it passes.** No perf gate; speed/efficiency reported beside correctness | **YES it "passes" correctness** — but ViperBench's job is to *measure & report* the gap (`slow_kernels.csv`), exposing what the others would have admitted |
| **Reward-hacking prevention** | **Partial:** all-trials `allclose` over 5 random seeds + adversarial suite (cache-reuse, input-mod, stream-timing); excessive-speedup (>10×) **warns only**, "experimental" | Multi-branch correctness; cold-cache timing; no published anti-cheat taxonomy `[UNVERIFIED]` | Unified-API contract + shared harness; reference *raises* on unsupported args. Hand-written kernels, not a generation target → reward-hacking largely N/A |

---

## 4. Analysis (draft prose for the RQ1 section)

**The gate is correctness; performance is, at best, a leaderboard coordinate.** Both prevailing benchmarks make
*functional correctness* the only thing a kernel must clear to be an accepted submission. KernelBench encodes
this literally: a kernel that matches the reference on five random-seed trials within per-dtype tolerance is
"correct," and the benchmark then *reports* a speedup rather than *requiring* one — `fast_0` is just the
correctness rate, and the vendored evaluator in this repo returns success (`exit 0`) on correctness alone,
regardless of speed (`AKO4ALL/bench/kernelbench/bench.py:1009`). TritonBench is the same shape: operators need
only execute correctly to be scored, after which speedup and GPU-efficiency are tabulated. The consequence is
exactly the RQ1 claim: **a functionally-correct DSL kernel can pass these benchmarks while being far slower
than the vendor library**, because nothing in the protocol turns "slow" into "fail."

**Where KernelBench is stronger than a strawman — and where it still leaks.** In fairness, KernelBench does
score speed (`fast_1`/`fast_2` demand a correct kernel that beats PyTorch, optionally `torch.compile`), and it
actively defends against several known reward-hacks via all-trials correctness over random seeds plus a
committed adversarial test suite (cache-reuse, input-modification, non-default-stream timing). But the
performance signal is **PyTorch-relative on a single hardcoded shape and fp32 by default**, and the one guard
aimed at "too-good-to-be-true" results — the >10× excessive-speedup check — is annotated "experimental" and
only emits a warning; it never fails a kernel. That loophole is not hypothetical: the Sakana "AI CUDA Engineer"
episode produced widely-cited 10–150× "speedups" that were evaluation exploits (memory reuse, dropped
computation) before being walked back — direct evidence that a correctness-gated, warn-only protocol "only
partially prevents reward hacking" (`PIVOT_FRAMING.md`).

**TritonBench actually points at the right fix — but narrowly.** TritonBench is the more performance-serious of
the two on one axis: it reports **GPU efficiency as a fraction of the A100's theoretical peak**, a
*baseline-independent roofline anchor* rather than a self-referential speedup. This is essentially the
non-circular signal our RQ3 heuristics argue for, so it is partial corroboration of our thesis, not merely a
target. The catch is breadth: TritonBench is **Triton-only, single-A100**, its speedup bar is the
reference/PyTorch op rather than a tuned vendor library, and — crucially — its efficiency number is *reported,
not gated*. So it still admits slow-but-correct kernels, and it says nothing about TileLang or other DSLs, other
GPUs, or controlled dtype/shape sweeps.

**Narrow coverage compounds the weak gate.** Both benchmarks under-cover the axes along which DSL kernel
quality actually varies. KernelBench fixes one shape per task (Level-1 matmul is always `4096²`) and defaults to
fp32; TritonBench evaluates on a single A100 with per-operator native dtypes. Neither systematically sweeps
shape, dtype, or hardware — yet our gap data show DSL kernels that look fine at one size collapse at another,
that the fp16/bf16 tensor-core regime is where DSLs most diverge from the vendor library, and that "efficient on
A100" does not transfer across Ada/H100/GH200. A benchmark that pins one shape, one dtype, and one GPU cannot
distinguish a robust kernel from one hyper-fit to its single test point.

**Why this motivates the pivot (and ViperBench's role).** The combination — a correctness-only hard gate, a
performance signal that is either reported-not-required or self-referential, partial anti-cheat, and narrow
shape/dtype/hardware coverage — is precisely why a comprehensive "is this kernel good?" benchmark is
combinatorially infeasible (DSL × dtype × shape × GPU × baseline). ViperBench is included above **only as a
diagnostic probe**: by holding one API across PyTorch/Triton/TileLang and *measuring and reporting* the gap
(`slow_kernels.csv`) across 22 kernels, 3 dtypes, multiple shapes, and 4–5 GPUs, it makes visible the exact
kernels that KernelBench/TritonBench would have stamped "correct." It is **not** offered as the missing
comprehensive benchmark — the paper's position is that no such benchmark is practical, which is what motivates
the lightweight, partly baseline-independent evaluation heuristics (comparability screen + roofline anchor) in
RQ3.

---

## Sources

**KernelBench — primary**
- Blog: <https://scalingintelligence.stanford.edu/blogs/kernelbench/>
- Paper page / PDF (arXiv:2502.10517): <https://scalingintelligence.stanford.edu/pubs/kernelbench/> · <https://scalingintelligence.stanford.edu/pubs/kernelbench.pdf>
- GitHub repo: <https://github.com/ScalingIntelligence/KernelBench>
- `src/kernelbench/eval.py` — correctness `torch.allclose(... atol=tolerance, rtol=tolerance)`; `get_tolerance_for_precision` lines 93–100 (fp32 1e-4, fp16/bf16 1e-2); `num_correct_trials` default 1 / `num_perf_trials` default 10; `check_for_excessive_speedup=True`, `excessive_speedup_threshold=10` (lines 410–412) warns only (lines 691–692): <https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/eval.py>
- `src/kernelbench/score.py` — `fastp`: `speed_up = baseline/actual; fast_p = sum(speed_up > p)/n` (lines 28–36): <https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/score.py>
- Adversarial unit tests (cache-reuse, input-modification, non-default-stream): <https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/unit_tests/test_eval_adversarial.py>
- Example single-shape problem (`N=4096`, fp32): <https://github.com/ScalingIntelligence/KernelBench/blob/main/KernelBench/level1/1_Square_matrix_multiplication_.py>
- Committed eager + `torch.compile` baseline timings: `results/timing/<gpu>/{baseline_time_torch.json, baseline_time_torch_compile_inductor_default.json}` (via repo tree)
- HuggingFace dataset: <https://huggingface.co/datasets/ScalingIntelligence/KernelBench>
- Fall-2025 roadmap (DSL/AMD backend expansion), Issue #74: <https://github.com/ScalingIntelligence/KernelBench/issues/74>

**KernelBench — reward-hacking episode (Sakana "AI CUDA Engineer")**
- TechCrunch (walk-back of speedup claims): <https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/>
- Sakana "Robust KBench" paper: <https://pub.sakana.ai/static/paper.pdf> · repo: <https://github.com/SakanaAI/robust-kbench>
- Memory-reuse exploit / dropped-conv example (miru): <https://x.com/miru_why/status/1892703900425486539>

**TritonBench — primary**
- arXiv abstract / HTML / PDF (2502.14752): <https://arxiv.org/abs/2502.14752> · <https://arxiv.org/html/2502.14752v1>
- ACL Findings 2025: <https://aclanthology.org/2025.findings-acl.1183/>
- (Distinct) Meta performance suite of the same name: <https://github.com/meta-pytorch/tritonbench>

**Repo (vendored evidence)**
- `AKO4ALL/bench/kernelbench/bench.py` — tolerances 259–265 (fp32 1e-3, fp16/bf16 5e-2); correctness all-trials 485–488; excessive-speedup warn-only 725–734; `get_inputs`/`get_init_inputs` override via `prepare_solution_source` 785–805; exit-code = correctness only 1009.
- `AKO4ALL/bench/kernelbench/GUIDE.md` — tolerance table 67–75 (fp32 1e-4, fp16/bf16 1e-2) — **diverges from the code above**.
- `ViperBench/` — 22 kernel dirs × {pytorch,triton,tilelang}_impl.py; per-kernel multi-shape `test.py` (e.g. `matmul/test.py` 5 shapes 64→4096; `layer_norm/test.py` 6 shapes incl. 3-D); GPU-namespaced `results/profile.<gpu>.csv` (RTX4000Ada, A100-PCIE-40GB, A100-SXM4-40GB, H100-80GB-HBM3); `results/slow_kernels.csv`.

**`[UNVERIFIED]` flags (double-check before citing in the paper)**
1. TritonBench **dtype coverage** — not enumerated in the paper text retrieved; treat as "per-operator native dtype, not a controlled axis."
2. TritonBench **published anti-reward-hacking measures** — none surfaced in the fetched sections; absence-of-evidence, not confirmed absence.
3. A secondary write-up's claim that **KernelBench uses 3–5 shapes/problem with geomean** is **contradicted** by the actual problem files (single hardcoded shape) — do **not** repeat it; flagged here so it isn't reintroduced.
