# ICSE Research Track #1787 — Artifact

> **How do you know a DSL GPU kernel is fast?** A Triton or TileLang kernel that is
> *functionally correct* can still run 5–1300× slower than the vendor library it is meant to
> replace — and pass the correctness-based benchmarks (KernelBench, TritonBench) used to gate
> DSL and LLM-generated kernels. This artifact studies that **correctness–performance gap as an
> evaluation problem**: we show where existing benchmarks miss it, why the gap arises, and how to
> detect it without exhaustive benchmark coverage.

This repository is the supporting artifact for the paper. It reproduces every table and headline
number, and contains the benchmark suite, the agentic kernel-optimization loop, and all root-cause
and heuristic experiments. (The paper PDF is provided separately through the review system.)

*Prepared for double-blind review.*

---

## Start here — three paths for reviewers

This artifact is organized so you can walk it in whichever order suits your review. Pick a path:

| # | Goal | Where to go | Time |
|---|------|-------------|------|
| **1** | **Read the code** — see what was built | [Repository structure](#repository-structure) → the three subsystems; then [Paper → artifact map](#paper--artifact-map) to jump from any claim to its code | browse |
| **2** | **Understand the code** — how each subsystem works and why | [Documentation index](#documentation-index-understand-the-code) — one purpose-written doc per subsystem | browse |
| **3** | **Reproduce the results** — run it yourself | [Reproducing results](#reproducing-results) — tiered from a 2-minute smoke test to the full locked-clock pipeline | minutes → hours |

If you only do one thing: run the **[smoke test](#tier-0--smoke-test-2-minutes-no-root)**
(one kernel, correctness + timing, ~2 min) to confirm the environment, then read the
**[paper at a glance](#the-paper-at-a-glance)** to see the three claims and where each is proven.

---

## What this artifact contains

The paper's argument spans three coupled subsystems, each with a distinct role:

- **`ViperBench/`** — the benchmark suite. 22 deep-learning kernels × 3 backends
  (**PyTorch** reference, **Triton**, **TileLang**), checked for correctness and timed for
  latency/peak-memory. This is the *probe* that exposes the hidden gap (RQ1/RQ2 evidence).
- **`experiments/`** — the analysis harness. The **RQ1** benchmark survey + "passes-but-slow"
  demonstration, the **RQ2** Nsight Compute root-cause sweep, and the **RQ3** roofline/comparability
  heuristic validation. GPU-portable: every output is namespaced under `results/<gpu_slug>/`.
- **`AKO4ALL/`** — the agentic kernel-optimization loop. A coding agent iteratively rewrites a
  single kernel for speed under a strict, anti-reward-hacking protocol. Source of the **RQ3**
  optimization-pattern evidence (the recovered kernels).

---

## The paper at a glance

The study is organized around three research questions. Each has one headline result and a
clear evidence base you can inspect or re-run.

**RQ1 — The evaluation gap.** *Do existing DSL/LLM-kernel benchmarks distinguish efficient
kernels from performance-poor ones?*
→ **No.** An idiomatic TileLang LayerNorm kernel **passes KernelBench's correctness gate (5/5
trials) yet runs 1293× slower** than PyTorch on the GH200. Correctness-based gating admits severe
slowdowns.
Evidence: [`experiments/RQ1_benchmark_survey.md`](experiments/RQ1_benchmark_survey.md) (KernelBench
& TritonBench coverage/anti-cheat analysis) + [`passes_but_slow.csv`](experiments/results/NVIDIA_GH200_480GB/passes_but_slow.csv).

**RQ2 — The hidden gap and its causes.** *How large and structured is the gap, and what causes it?*
→ The causes **differ by kernel family**. TileLang normalization/reduction slowdowns are mostly
*repairable authoring defects* (sequential `T.serial` reductions, unnecessary dtype conversions);
convolution and large GEMM retain *residual* gaps from code generation, autotuning coverage, and
vendor-library algorithm selection.
Evidence: the 22-kernel suite (`ViperBench/results/`) + the Nsight Compute counter sweep
(`experiments/results/<gpu>/ncu_summary.csv` + [`NCU_FINDINGS.md`](experiments/results/NVIDIA_GH200_480GB/NCU_FINDINGS.md)).

**RQ3 — Guidance without a comprehensive benchmark.** *Can lightweight heuristics flag poor
kernels and guide fixes?*
→ **Yes.** Two complementary screens — **library-relative efficiency** (`E_lib`) and **roofline
utilization** — discriminate efficient from inefficient kernels, and a small set of recurring
optimization patterns (dominant fix: `T.serial → T.reduce` + native-dtype I/O) recovers up to
**1224×**.
Evidence: [`experiments/exp_cliff_roofline.py`](experiments/exp_cliff_roofline.py) →
[`cliff_roofline.csv`](experiments/results/NVIDIA_GH200_480GB/cliff_roofline.csv) (the roofline
anchor, = `tab:roofline`) + the five optimization campaigns in
[`AKO4ALL/results/`](AKO4ALL/results/optimization_results.csv).

> **`E_lib`** = library efficiency = (baseline latency ÷ DSL latency) × 100%. PyTorch /
> cuBLAS / cuDNN = 100%; higher is better; <100% means slower than the vendor library.

---

## Repository structure

```
ASE-GPUDSL-ARTIFACT/
├── ViperBench/                       # (1) Benchmark suite — 22 kernels × {PyTorch, Triton, TileLang}
│   ├── <kernel>/                     #     pytorch_impl.py · triton_impl.py · tilelang_impl.py · test.py
│   ├── run_all.py                    #     correctness runner (all 22 kernels)
│   ├── benchmark*.py                 #     latency + peak-memory sweep (writes results/profile.csv)
│   ├── tuning/                       #     per-arch autotuning sweep (tuning_cache.json)
│   └── results/
│       ├── profile.<gpu>.csv         #     per-arch latency/memory (A100-SXM4, GH200, A100-PCIE, H100)
│       └── slow_kernels.csv          #     derived: kernels where DSL < PyTorch at large shape
├── experiments/                      # (2) Analysis harness (RQ1 survey, RQ2 ncu, RQ3 heuristics)
│   ├── RQ1_benchmark_survey.md       #     RQ1 — KernelBench & TritonBench coverage/anti-cheat survey
│   ├── exp_passes_but_slow.py        #     RQ1 — "correct yet slow" demonstration
│   ├── ncu_counters.sh               #     RQ2 — Nsight Compute counter sweep driver
│   ├── consolidate_ncu.py            #     RQ2 — roll up ncu CSVs → ncu_summary.csv
│   ├── exp_cliff_roofline.py         #     RQ3 — roofline + comparability heuristic validation
│   ├── exp_*.py                      #     supporting root-cause experiments (fp32 GEMM, conv, winograd…)
│   ├── repro/                        #     locked-clock reproduction pipeline (see Tier 3)
│   ├── MACHINE_SETUP.md              #     host / driver / ncu setup
│   └── results/
│       ├── NVIDIA_A100-SXM4-40GB/    #     primary evidence (Ampere, sm_80)
│       ├── NVIDIA_GH200_480GB/       #     primary evidence (Grace Hopper, sm_90)
│       ├── NVIDIA_A100-PCIE-40GB/    #     form-factor control
│       ├── NVIDIA_H100_80GB_HBM3/    #     cross-family control
│       └── cross_arch/               #     aggregated cross-arch tables + generator
└── AKO4ALL/                          # (3) Agentic kernel-optimization loop
    ├── README.md                     #     how the optimizer works
    ├── TASK.md · HINTS.md            #     the protocol + constraints the agent obeys
    ├── prepare_kernel.py             #     ViperBench kernel → KernelBench-format export
    └── results/
        ├── optimization_results.csv  #     the 5 completed campaigns (feeds tab:mitigation)
        └── optimized/                #     optimized source + per-iteration logs
```

---

## Documentation index (understand the code)

One purpose-written document per subsystem — start here to understand *how* and *why*, not just
*what*:

| Subsystem | Read this | Covers |
|---|---|---|
| ViperBench | the [ViperBench section below](#viperbench-the-benchmark-suite) | The 4-file-per-kernel contract, unified-API invariant, import-time tuning |
| RQ1 evidence | [`experiments/RQ1_benchmark_survey.md`](experiments/RQ1_benchmark_survey.md) | Why KernelBench/TritonBench miss performance-poor kernels |
| RQ2 evidence | [`experiments/results/<gpu>/NCU_FINDINGS.md`](experiments/results/NVIDIA_GH200_480GB/NCU_FINDINGS.md) | Counter-grounded root-cause interpretation per kernel |
| AKO4ALL | [`AKO4ALL/README.md`](AKO4ALL/README.md) + [`TASK.md`](AKO4ALL/TASK.md) | The optimization protocol, anti-reward-hacking design |
| Cross-arch | [`experiments/MACHINE_SETUP.md`](experiments/MACHINE_SETUP.md) + the `experiments/repro/` scripts | Replaying the full suite on a new GPU |
| Cross-arch data | [`experiments/results/cross_arch/README.md`](experiments/results/cross_arch/README.md) | The aggregated cross-architecture tables + generator |

### ViperBench: the benchmark suite

Each kernel lives in `ViperBench/<kernel>/` with exactly four files:

| File | Role |
|------|------|
| `pytorch_impl.py`  | reference (torch / cuBLAS / cuDNN) — the correctness golden |
| `triton_impl.py`   | hand-written Triton kernel |
| `tilelang_impl.py` | hand-written TileLang kernel |
| `test.py`          | wires all three into the shared harness and `sys.exit(0/1)`s |

**Unified-API contract (the key invariant):** all three `*_impl.py` files export a function with
the **same name as the kernel directory** and the **same signature**, so the three backends are
drop-in interchangeable (e.g. every backend defines `layer_norm(x, weight, bias, eps=1e-5)`). The
shared harness (`ViperBench/test_utils.py`) compares each DSL backend against the PyTorch golden at
per-dtype tolerances (fp32 `1e-5`, fp16 `1e-3`, bf16 `1e-2`).

---

## Paper → artifact map

Use this to jump from any claim, table, or figure in the paper to the code and data behind it.

### Sections → evidence

| Paper section | Evidence in this repo |
|---|---|
| §4 The Evaluation Gap (**RQ1**) | `experiments/RQ1_benchmark_survey.md` + `exp_passes_but_slow.py` → `results/<gpu>/passes_but_slow.csv`; suite coverage in `ViperBench/` |
| §5 Root Causes of the Hidden Gap (**RQ2**) | `ViperBench/results/profile.<gpu>.csv` + `slow_kernels.csv`; `experiments/ncu_counters.sh` → `results/<gpu>/ncu_summary.csv` + `NCU_FINDINGS.md`; supporting `exp_fp32_gemm.py`, `exp_conv_filters.py`, `exp_winograd_isolation.py` |
| §6 Guidance Without a Comprehensive Benchmark (**RQ3**) | `experiments/exp_cliff_roofline.py` → `results/<gpu>/cliff_roofline.csv`; `AKO4ALL/results/optimization_results.csv` + `results/optimized/<kernel>_iterations.md` |

### Tables & figures → data

| Label | Source data |
|---|---|
| `tab:gemm` / `tab:conv` / `tab:norm` | `ViperBench/results/slow_kernels.csv` (+ `profile.<gpu>.csv` for small configs) |
| `tab:summary` | Medians over the per-category cells of the three tables above |
| `tab:rootcauses` | `experiments/results/<gpu>/ncu_summary.csv` + `NCU_FINDINGS.md` |
| `tab:roofline` | `experiments/results/<gpu>/cliff_roofline.csv` (from `exp_cliff_roofline.py`) |
| `tab:mitig:norm` / `tab:mitig:conv` / `tab:mitigation` | `AKO4ALL/results/optimization_results.csv` (+ `experiments/results/<gpu>/conv_mitigation*.csv` for the conv arm) |
| `fig:overview` | Numbers traceable to `ViperBench/results/slow_kernels.csv` via the evaluation tables |

---

## Reproducing results

Reproduction is tiered so you can go as deep as you have time and hardware for. **Tiers 0–2 need
only a CUDA GPU** (no root); **Tier 3** produces the paper's authoritative locked-clock numbers and
needs `sudo`. Every step needs a CUDA GPU except the RQ1 benchmark survey (`experiments/RQ1_benchmark_survey.md`), which is a desk analysis you can read without hardware.

### Environment & install

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
# Pins: torch==2.8.0+cu126, triton==3.4.0, tilelang==0.1.6.post1
# Nsight Compute (`ncu`) is additionally required for the RQ2 counter sweep (Tier 2).
```

The PyTorch cu126 wheel ships its own CUDA runtime; the host only needs a recent NVIDIA driver
(≥ 555). Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Tier 0 — Smoke test (2 minutes, no root)

Confirm the toolchain end-to-end on one kernel before committing to a full run:

```bash
python ViperBench/layer_norm/test.py        # correctness: PyTorch vs Triton vs TileLang → exit 0 = pass
python ViperBench/run_all.py                 # all 22 kernels, prints a correctness summary
```

### Tier 1 — Reproduce the headline results, by RQ

**RQ1 — the evaluation gap (the "passes-but-slow" demonstration).** Runs naive DSL kernels through
KernelBench-style correctness gating and shows they pass yet are 5–1300× slow:

```bash
python experiments/exp_passes_but_slow.py    # → experiments/results/<gpu_slug>/passes_but_slow.csv
```

The benchmark-survey half of RQ1 is a desk analysis in
[`experiments/RQ1_benchmark_survey.md`](experiments/RQ1_benchmark_survey.md) (no run needed).

**RQ3 — the heuristic validation (roofline + comparability).** Times naive vs optimized vs library
per kernel and computes `E_lib` and roofline fraction (this is `tab:roofline`):

```bash
python experiments/exp_cliff_roofline.py     # → experiments/results/<gpu_slug>/cliff_roofline.csv
```

The optimization-pattern half of RQ3 is the five completed AKO4ALL campaigns, already recorded in
[`AKO4ALL/results/optimization_results.csv`](AKO4ALL/results/optimization_results.csv) with full
per-iteration trajectories in `AKO4ALL/results/optimized/<kernel>_iterations.md`. To re-run a
campaign yourself, see [`AKO4ALL/README.md`](AKO4ALL/README.md).

### Tier 2 — Full benchmark suite + RQ2 root-cause sweep

```bash
# Latency + peak-memory sweep. NOTE: these OVERWRITE ViperBench/results/profile.csv in place
# (they are NOT per-arch). Use repro/regen_profile.sh (Tier 3) to get a named profile.<gpu>.csv.
python ViperBench/benchmark.py               # PyTorch + Triton
python ViperBench/benchmark_tilelang.py      # merge TileLang into profile.csv
python ViperBench/benchmark_tuned.py         # add *_tuned variants

# Per-arch autotuning sweep (writes results/tuning_cache.json) — run as a module from ViperBench/
cd ViperBench && python -m tuning.sweep --all && cd ..

# RQ2 — Nsight Compute counter sweep → root-cause table (tab:rootcauses)
# ncu needs elevated GPU-counter access: run under sudo (or set NVIDIA's
# RmProfilingAdminOnly=0). See experiments/MACHINE_SETUP.md.
bash experiments/ncu_counters.sh             # per-kernel counters → results/<gpu>/ncu/*.csv
python experiments/consolidate_ncu.py        # roll up → results/<gpu>/ncu_summary.csv
```

### Tier 3 — Authoritative locked-clock numbers (needs `sudo`) & cross-arch replay

The numbers reported in the paper are measured with GPU clocks **locked** to a sustained value to
remove run-to-run throttling noise. This requires `nvidia-smi -lgc/-lmc` (root):

```bash
export PYTHON=$(command -v python)                     # ← REQUIRED: repro scripts default to a
                                                        #   hardcoded venv path; override it first
bash experiments/repro/lock_clocks.sh                  # discovery: find this GPU's SUSTAINED clock
bash experiments/repro/run_pipeline.sh <GR_MHZ> <MEM_MHZ>   # full locked pipeline for one GPU
#   e.g. A100-SXM4: run_pipeline.sh 1215 1215     GH200: run_pipeline.sh 1320 2619
bash experiments/repro/regen_profile.sh --tuned        # per-arch profile.<gpu>.csv (snapshots+restores profile.csv)
```

To replay the entire suite on a **different GPU**, use the scripts in `experiments/repro/` (with
`experiments/MACHINE_SETUP.md` for host/driver/`ncu` setup). Results auto-namespace under
`experiments/results/<gpu_slug>/`, so existing evidence is never overwritten.

### Reproduction notes (read before running Tier 3)

- **`PYTHON` must be set.** Scripts under `experiments/repro/` default their interpreter to a
  hardcoded venv path (`/home/ubuntu/dslperf-venv/bin/python`). Always `export PYTHON=$(command -v python)`
  (or your venv's python) first, or they will fail to find the interpreter.
- **`ViperBench/results/profile.csv` is overwritten in place** by `benchmark*.py` — it is *not*
  per-architecture. The committed `profile.<gpu>.csv` files are produced by `repro/regen_profile.sh`,
  which snapshots and restores the live `profile.csv` around the run.
- **`sudo` is only needed for clock locking and `ncu`** (Tier 3 + the RQ2 sweep). Everything in
  Tiers 0–2 runs as an unprivileged user; those numbers will carry more run-to-run variance than the
  locked-clock ones in the paper.
- **`experiments/` outputs are GPU-namespaced** (`results/<gpu_slug>/…`), so running on a new GPU is
  non-destructive.

---

## Reference

Supporting specifications for the measurements above.

### Kernel suite provenance

The 22-kernel suite covers five operator categories. Triton implementations are drawn from
TritonBench except `layer_norm` (TorchInductor reference); all TileLang implementations are custom
re-implementations of the same operators behind the same interface.

| Category | Kernels |
|---|---|
| GEMM | `matmul`, `batched_matmul`, `linear_activation` |
| Attention | `attention` |
| Convolution | `conv2d` |
| Normalization | `layer_norm`, `rms_norm` |
| Element-wise / Reduction | `add`, `mul`, `relu`, `leaky_relu`, `softmax`, `log_softmax`, `logsumexp`, `swiglu`, `argmax`, `max_reduction`, `mean_reduction`, `cross_entropy`, `matrix_transpose`, `index_select`, `embedding` |

### Baseline specifications

| Kernel category | Baseline |
|---|---|
| GEMM | cuBLAS via `torch.matmul` |
| Convolution | `nn.Conv2d` (NCHW, default cuDNN algorithm selection) |
| Normalization | `F.layer_norm` / `F.rms_norm` |
| Element-wise / Reduction | PyTorch eager |
| Attention | `F.scaled_dot_product_attention` with FlashAttention backend |

### Measurement protocol

**Timing.** Host-synchronized GPU time via `time.perf_counter()` bracketed by
`torch.cuda.synchronize()`; 10 warm-up + 100 timed iterations, median reported, each workload in
isolation. Run-to-run relative std-dev = 0.0–0.9% on locked clocks.

**Clock locking.**

| GPU | Graphics lock | Memory lock |
|---|---|---|
| A100-SXM4-40GB | 1215 MHz | 1215 MHz |
| GH200 | 1320 MHz | 2619 MHz |

**Nsight Compute counters.** Seven counters per kernel from a single `ncu` execution (verified
within 2% across five repeats): global-load efficiency, sectors/request, registers/thread, register
spills, long-scoreboard stalls, barrier stalls, L2 sector hit rate.

**KernelBench evaluator (RQ1).** The "passes-but-slow" demonstration invokes KernelBench's
`eval_kernel_against_ref`, which overrides candidate `get_inputs`/`get_init_inputs` with the
reference's, checks output equivalence at the same tolerances as our suite, and warns *only* on >10×
speedup (thus admitting any slowdown). See `experiments/results/<gpu>/passes_but_slow.csv`.

### Hardware and software

**Primary GPUs**

| Property | A100-SXM4-40GB | GH200 |
|---|---|---|
| Architecture | Ampere (`sm_80`) | Grace Hopper (`sm_90`) |
| SMs | 108 | 132 |
| Memory | 40 GB HBM2e | 96 GB HBM3 |
| Peak mem. BW | ~1.5 TB/s | ~4.0 TB/s |
| Peak FP16 TC | ~312 TFLOP/s | ~989.5 TFLOP/s |
| L2 cache | 40 MB | 60 MB |

**Cross-architecture controls:** A100-PCIE-40GB (form-factor control, `sm_80`);
H100-80GB-HBM3 (cross-family control, `sm_90`).

**Software:** PyTorch 2.8.0+cu126 · Triton 3.4.0 · TileLang 0.1.6.post1 · cuDNN (bundled with
PyTorch) · Nsight Compute 2026.2.0.

### Cross-architecture generalization

Median `E_lib` (%) per operator category, untuned defaults. The qualitative pattern (GEMM/element-wise
competitive; convolution and TileLang normalization severely behind) holds across all three GPUs shown.

| Category | Triton SXM4 | Triton PCIE | Triton H100 | TileLang SXM4 | TileLang PCIE | TileLang H100 |
|---|--:|--:|--:|--:|--:|--:|
| GEMM | 77.3% | 75.5% | 68.4% | 69.4% | 60.3% | 69.2% |
| Convolution | 31.6% | 38.0% | 37.7% | 9.5% | 11.1% | 7.8% |
| Normalization | 121.5% | 137.3% | 130.5% | 1.0% | 0.8% | 0.9% |
| Element-wise / Red. | 65.3% | 73.0% | 72.9% | 51.0% | 55.3% | 60.9% |

Full data and generator: [`experiments/results/cross_arch/`](experiments/results/cross_arch/).

### Key optimization results (RQ3)

Five kernel-optimization campaigns (AKO4ALL). One iteration = one source edit + one benchmark run +
correctness check + git commit.

| Kernel | DSL | Input | Before | After | Speedup | Dominant pattern |
|---|---|---|--:|--:|--:|---|
| layer_norm | TileLang | (8192, 8192) bf16 | 1090.0 ms | 0.89 ms | **1224×** | `T.serial → T.reduce` + native bf16 I/O |
| rms_norm | TileLang | (8192, 8192) fp16 | 716.0 ms | 0.90 ms | **796×** | `T.serial → T.reduce` + native fp16 I/O |
| argmax | TileLang | (8192, 32768) fp16 | 16.2 ms | 1.75 ms | **9.26×** | tiled shared-mem loads + native fp16 I/O |
| conv2d | Triton | (32, 256, 128, 128) fp16 | 32.1 ms | 12.5 ms | **2.57×** | implicit GEMM + fp16 tensor cores + autotune |
| matmul | Triton | (4096, 4096) fp16 | 2.71 ms | 1.63 ms | **1.66×** | `@triton.autotune` + `GROUP_SIZE_M` L2 swizzle |

Source: [`AKO4ALL/results/optimization_results.csv`](AKO4ALL/results/optimization_results.csv);
per-iteration logs in `AKO4ALL/results/optimized/<kernel>_iterations.md`.
