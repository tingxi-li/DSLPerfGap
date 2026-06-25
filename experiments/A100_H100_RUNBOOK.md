# A100 / H100 Replay Runbook — ASE-2026 #4134 Cross-Architecture Validation

This is the **"later" plan**: replay the entire rebuttal experiment suite on an
A100 and an H100 to test whether the root-cause taxonomy *generalizes across
architectures* (**R1-Q5, R3, W5**; Experiment 6 in `REBUTTAL_PROTOCOLS_CRITICAL.md`).

It is designed to be **turnkey**: the harness (`experiments/_harness.py`) queries
device properties at runtime and tags every result by GPU, so the *identical
code* that runs on RTX 4000 Ada now runs unchanged on A100/H100 and writes to a
**new** `experiments/results/<gpu_slug>/` directory — Ada/A100/H100 outputs never
collide. No source edits are required for a new GPU.

> **Why this can't be reasoned, only run:** A100/H100 differ exactly where the
> taxonomy lives — HBM2e/HBM3 at **1.5–3.3 TB/s** vs Ada's **360 GB/s** GDDR6,
> larger L2, and (H100) **TMA + wgmma** async-pipeline hardware cuBLAS exploits.
> RC2b (L2/BW-bound) and the GEMM gap may shrink or shift; the RC0 TileLang
> LayerNorm anomaly (a sync/authoring bug) should reproduce arch-independently.

> **NOTE (revision, 2026-06):** the study now treats **A100-SXM4-40GB as the PRIMARY
> GPU** (all numbers locked-clock), with **H100 + A100-PCIE** as cross-architecture
> generalization points. Run section **(0)** on the H100 box to produce the *same*
> artifact set under the *same* locked-clock methodology so everything stays consistent.
> The committed H100/A100-PCIE replays were collected **un-locked** earlier (noisier at
> sub-ms shapes); re-running section (0) under locked clocks supersedes them.

---

## (0) Quick-start — reproduce the full primary artifact set on a new GPU

Helper scripts live in **`experiments/repro/`**. Full sequence (one GPU, ~2–3 GPU-hours):

```bash
export PYTHON=/home/ubuntu/dslperf-venv/bin/python    # env with torch 2.8 + triton 3.4 + tilelang 0.1.6

# 1. Find the SUSTAINED locked clock (NOT the max — see gotchas).
bash experiments/repro/lock_clocks.sh                 # load-tests max; prints the sustained GR/MEM to use

# 2. Counter + timing + significance pipeline at the sustained clock.
bash experiments/repro/run_pipeline.sh <GR> <MEM>     # A100-SXM4 used: 1215 1215
#    -> results/<slug>/{run_all.log, fp32_gemm.csv, conv_*.csv, fused_baselines.csv,
#       winograd_isolation.csv, autotune_matmul.csv, correctness_edge.csv,
#       ncu/, ncu_summary.csv, significance.csv, clock_lock.txt}

# 3. Regenerate this arch's ViperBench profile (untuned + tuned sweep).
bash experiments/repro/lock_clocks.sh <GR> <MEM>      # re-lock (pipeline resets clocks at the end)
bash experiments/repro/regen_profile.sh --tuned       # -> ViperBench/results/profile.<short>.csv (+ .tuned.csv)

# 4. Re-time every mitigation (AKO4ALL norm/argmax/matmul + the optimized reduction/softmax family).
$PYTHON experiments/repro/retime_mitigation.py        # -> results/<slug>/mitigation_retime.csv

# 5. Reset clocks (MANDATORY — else the GPU is left pinned).
sudo nvidia-smi -i 0 -rgc ; sudo nvidia-smi -i 0 -rmc
```

### Gotchas that WILL bite you (learned the hard way on A100-SXM4)
- **Clock lock ≠ max clock.** Locking the GPU's *max* graphics clock is silently
  power-capped under sustained tensor-core load (throttle bit `0x4` = SW Power Cap),
  giving a *fluctuating* clock — the opposite of locking. `lock_clocks.sh` load-tests
  and reports the **sustained** value. A100-SXM4 (400 W): 1410 caps to ~1215 → lock
  **1215/1215**. H100 (700 W): query first — 1980 may hold (more headroom) or may cap.
- **Issue `-lgc` and `-lmc` SEPARATELY.** Driver ≥ 610 rejects the combined
  `nvidia-smi -lgc X -lmc Y` ("Only one device modification may be done at a time").
- **`ncu` is admin-gated** (`/proc/driver/nvidia/params` → `RmProfilingAdminOnly: 1`).
  `ncu_counters.sh` runs ncu under **`sudo`** (root bypasses the gate; no reboot needed).
  `ncu` is off `PATH` at `/usr/local/cuda/bin/ncu` (scripts use `$NCU`).
- **TileLang JIT needs the venv's bundled CUDA libs on `LD_LIBRARY_PATH`**
  (`libnvrtc.so.12`, `libcudart.so.12`). `retime_mitigation.py` sets this itself; if you
  import `experiments/opt_kernels/*` elsewhere, prepend the venv `nvidia/*/lib` dirs.
- **`ViperBench/results/profile.csv` is NOT arch-namespaced** — `benchmark*.py` overwrite
  it in place. `regen_profile.sh` snapshots+restores it and writes `profile.<short>.csv`.
- **Two slug conventions (don't cross them):** `experiments/results/<slug>/` uses the
  `_harness` slug `NVIDIA_A100-SXM4-40GB`; the ViperBench profile uses the **short** name
  `profile.A100-SXM4-40GB.csv` (`H100-80GB-HBM3` for H100). The scripts derive both.

---

## (a) Get the repo onto the target machine

```bash
# Option 1: clone (preferred)
git clone <your-fork-or-origin> ASE-GPUDSL-ARTIFACT
cd ASE-GPUDSL-ARTIFACT

# Option 2: copy the working tree (rsync from the Ada box)
rsync -av --exclude '__pycache__' --exclude 'experiments/results' \
    <ada-host>:/home/lxt230026/ASE-GPUDSL-ARTIFACT/ ./ASE-GPUDSL-ARTIFACT/
```
You do **not** need to copy `experiments/results/` — A100/H100 write their own
`results/<gpu_slug>/` dirs. (Copying Ada's results is harmless; they won't
collide because the slug differs.)

## (b) Install the stack — NO code changes for the new SM

```bash
python -m pip install --upgrade pip
# Pinned install (reproduces the Ada toolchain on the new SM):
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```
- The repo-root `requirements.txt` pins `torch==2.8.0+cu126`, `triton==3.4.0`,
  `tilelang==0.1.6.post1` — the exact versions reported in
  `paper-latex-project/tex/methodology.tex:161`. A bare `pip install torch triton
  tilelang` would silently pull newer wheels and shift cuBLAS/cuDNN heuristics
  and the TileLang autotuner output, so cross-arch comparisons must use the pin.
- The PyTorch cu126 wheel ships its own bundled CUDA runtime; only the host
  driver and the system `ncu` need to match the new SM.
- **TileLang and Triton JIT-recompile for the target SM automatically.** First
  call on A100 (sm_80) / H100 (sm_90) triggers a fresh compile + autotune; the
  `run_one_kernel.py` warmup loop and `_harness.time_kernel` warmup absorb that
  compile so it is never timed or profiled.
- Match the CUDA toolkit to the driver. For **H100 (sm_90)** you need a
  CUDA 12.x toolchain and an `ncu` ≥ 2023.2 that knows `gh100`; CUDA 12.4+ /
  Nsight Compute 2024.x (what we used on Ada) covers both A100 and H100.
  Verify:
  ```bash
  nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader
  ncu --version | head -1     # must be >= 2023.2 for sm_90
  ```
- Sanity-check the portability layer (prints the *detected* GPU, no hardcoding):
  ```bash
  python experiments/_harness.py     # banner + device_info + a trivial matmul timing
  ```

### (b.1) Pre-flight smoke test (cheap, ~5 min — do this BEFORE the full suite)

Every `exp_*.py` accepts `--smoke` (forwarded by `run_all.sh`) for a tiny-input
plumbing check. Run it first as a toolchain sanity check on the rented GPU
*before* burning a full GPU-hour on a misconfigured stack:

```bash
bash experiments/run_all.sh --smoke    # ~5 minutes; writes <slug>/significance_smoke.csv etc.
```

If the smoke pass surfaces a Triton / TileLang JIT or import failure, stop and
fix it before the full run.

### (b.2) Preserve Ada CSVs that would be clobbered

`ViperBench/results/profile.csv` has no per-arch namespacing — re-running
`ViperBench/benchmark.py` on the new GPU **overwrites the Ada baseline in
place** (the experiments under `experiments/results/<gpu_slug>/` *do* auto-
namespace; this one CSV does not). Before any `ViperBench/benchmark*.py`
invocation on the new box:

```bash
cp ViperBench/results/profile.csv ViperBench/results/profile.RTX4000Ada.csv
cp ViperBench/results/slow_kernels.csv ViperBench/results/slow_kernels.RTX4000Ada.csv
```

Also do **NOT** invoke these two scripts on A100/H100:
- `ViperBench/benchmark_attn.py` — Ada-only `D=64` workaround; would corrupt
  the headline `D=128` attention rows in the new arch's `profile.csv`.
- `ViperBench/benchmark_fix.py` — historical one-off patch script; would
  overwrite three tilelang rows from a stale earlier run.

The live benchmark scripts to use are `benchmark.py`, `benchmark_tilelang.py`,
and `benchmark_tuned.py` only.

### (b.3) Decide the `tuning_cache.json` stance for the new arch

`ViperBench/results/tuning_cache.json` currently contains **only** RTX 4000 Ada
entries (keyed by `<kernel>/<impl>/<gpu_arch>` via `ViperBench/tuning/cache.py`).
On the new GPU, every kernel transparently falls back to `{}` and uses
hardcoded defaults — **correctness is unchanged, but the `*_tuned` rows in
`profile.csv` will be identical to the untuned rows**. Three options:

1. **Run the full sweep (recommended for headline numbers)** — multi-GPU-hour:
   ```bash
   cd ViperBench && python -m tuning.sweep --all
   ```
   Then re-run `benchmark_tuned.py`. The cache adds new
   `<kernel>/<impl>/<A100-or-H100-arch>` entries; Ada entries are untouched.
2. **Skip `benchmark_tuned.py` on the new arch** — only run `benchmark.py` and
   `benchmark_tilelang.py`. Document in the cross-arch comparison that the new-arch
   numbers are untuned defaults.
3. **Accept the unswept defaults** — run `benchmark_tuned.py` and note that
   `*_tuned == *` for the new arch.

The portable rebuttal experiments under `experiments/` do **not** depend on the
ViperBench tuning cache (they import the optimized kernels directly from
`AKO4ALL/results/optimized/` and use those kernels' own `@triton.autotune`
search), so this decision affects only the §5/§7.3 reconciliation rows.

## (c) Request profiling permissions (same blocker as Ada)

Hardware counters are gated by the **same** NVIDIA driver flag. On Ada it was
blocked (`RmProfilingAdminOnly: 1`). On the new machine:

```bash
# Check first (no sudo):
grep RmProfilingAdminOnly /proc/driver/nvidia/params      # 0 = unblocked, 1 = blocked

# If it reads 1, an admin runs (on a cloud ROOT instance you can do this yourself):
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
    | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo dracut -f            # RHEL/CentOS;  Debian/Ubuntu: sudo update-initramfs -u
sudo reboot               # or rmmod/modprobe the nvidia* modules if headless
# verify after reboot:
grep RmProfilingAdminOnly /proc/driver/nvidia/params      # must read 0
```
**Cloud tip:** on AWS p4d/p5, GCP a2/a3, Lambda, RunPod root instances you
typically have root, so this is a 3-line self-service step + one reboot. Request
it on **Day 0** of the GPU slot — it gates the entire counter table (Exp 1/5).

`experiments/ncu_counters.sh` pre-checks this flag and **exits 2 with the fix**
if still blocked, so you never burn GPU time dumping `ERR_NVGPUCTRPERM`.

## (d) Run the suite — results auto-land in a NEW per-arch dir

```bash
# 1. Timing + correctness suite (serialized on one pinned GPU):
bash experiments/run_all.sh                 # defaults to CUDA_VISIBLE_DEVICES=0
#    -> experiments/results/<A100_or_H100_slug>/*.csv + run_all.log
#
# This includes a 4-way conv sweep — {baseline, --mitigation} x {--shape small,
# --shape large} — that produces conv_{filters,mitigation}_{small,large}.csv,
# the four files cited by tab:mitig:conv / REVISION id=4134A-W6 in the paper.

# 2. Hardware counters (after permissions granted):
bash experiments/ncu_counters.sh
#    -> experiments/results/<A100_or_H100_slug>/ncu/*.csv

# 3. Roll up ncu/*.csv into the consolidated table the paper cites:
python experiments/consolidate_ncu.py
#    -> experiments/results/<A100_or_H100_slug>/ncu_summary.csv
#    Cited by paper-latex-project/tex/analysis.tex (tab:rootcauses) and
#    referenced in REVISION_TODO.md. Skip this step and the paper's RC table
#    has no per-arch evidence.
```
- The GPU slug (e.g. `NVIDIA_A100-SXM4-80GB`, `NVIDIA_H100_80GB_HBM3`) is
  derived from `torch.cuda.get_device_name`, so **A100/H100 will not collide
  with Ada** (`NVIDIA_RTX_4000_Ada_Generation`). Re-running overwrites only that
  arch's CSVs (idempotent).
- If the box has multiple GPUs, pin one: `CUDA_VISIBLE_DEVICES=1 bash experiments/run_all.sh`.
- Counter metric IDs in `ncu_counters.sh` are **arch-portable** — the same
  `smsp__`, `l1tex__`, `lts__`, `launch__`, `dram__` strings work on sm_80 / sm_89 /
  sm_90; no edits needed.
- TileLang `--kernel-name regex:func_kernel` filter in `ncu_counters.sh` is
  symbol-name-dependent. If the sm_90 lowering branch has shifted symbol
  naming, smoke-test with `ncu --list-launches python experiments/run_one_kernel.py
  layer_norm tilelang large` and update the regex if needed.

## (d.5) Locked-clock significance — RUN MANUALLY (needs sudo)

`exp_significance.py` is **not** in `run_all.sh` because it needs `sudo
nvidia-smi -lgc/-lmc`. It reports run-to-run dispersion (median + std + p95 +
significance verdict) on the near-parity kernels — the locked-clock evidence
the rebuttal `To-A #7` and `REVISION_TODO #6` pin the "94.6% vs 97.8%" framing
to.

The script auto-queries this GPU's `clocks.max.{gr,mem}` via `nvidia-smi`, so
no per-arch threshold is hardcoded; lock to those targets (or override with
`--lock-gr-mhz` / `--lock-mem-mhz` for sub-max headroom):

```bash
# Use experiments/repro/lock_clocks.sh to find the SUSTAINED clock (NOT the max --
# the max power-caps under load; see section (0) gotchas). It issues -lgc/-lmc
# SEPARATELY (driver >= 610 rejects the combined form). Verified sustained targets:
#   RTX 4000 Ada (sm_89, 130 W):   -lgc 1400  -lmc 9001  (gr holds; mem -> 8551 under load)
#   A100-SXM4-40GB (sm_80, 400 W): -lgc 1215  -lmc 1215  (1410 power-caps; 1215 holds flat)
#   H100-SXM5-80GB (sm_90, 700 W): query first -- 1980 may hold; lock_clocks.sh reports it
bash experiments/repro/lock_clocks.sh             # discovery -> prints sustained <GR> <MEM>
bash experiments/repro/lock_clocks.sh <GR> <MEM>  # lock at the sustained value

# Re-time the noise-sensitive set at the sustained clock (pass the same targets):
python experiments/exp_significance.py --lock-gr-mhz <GR> --lock-mem-mhz <MEM>
#    -> experiments/results/<gpu_slug>/significance.csv

# Reset clocks afterward (mandatory): SEPARATE commands.
sudo nvidia-smi -i 0 -rgc ; sudo nvidia-smi -i 0 -rmc
```

The script self-aborts with the GPU-specific lock recipe if measured clocks
are not within `--lock-tolerance-pct` (default 5%) of the target.

## (e) Minimal high-value subset (when GPU time is limited, ~1 GPU-day/arch)

If you can't run the full suite, this subset still answers the reviewers'
generalization question:

| Target | Why it matters | How |
|---|---|---|
| **matmul 16384²** | headline GEMM gap; RC2b L2 thrash; TMA/wgmma effect on H100 | `python experiments/run_one_kernel.py matmul triton large` (and `pytorch`, `tilelang`) + counter set D |
| **conv2d 3×3** `(32,256,128,128)` | headline conv gap; Winograd eligibility | `python experiments/run_one_kernel.py conv2d triton large` + sets A/B; `python experiments/exp_winograd_isolation.py` |
| **layer_norm 8192²** | the **RC0 anomaly** — memory-latency-bound under `T.serial`; expected to reproduce arch-independently | `python experiments/run_one_kernel.py layer_norm tilelang large` + sets A/C |
| **argmax + max_reduction** `(8192,32768)` | RC0a `warp_stall_long_scoreboard` (memory latency under `T.serial`; barrier ≈ 0) | `python experiments/run_one_kernel.py argmax tilelang large` (and `max_reduction`) + set C |
| **counter spot-check** | sets **A** (vectorized loads) + **C** (warp stalls) on the five above | `bash experiments/ncu_counters.sh` (override `TARGETS=...` to trim) |

Trim the counter run without editing the script:
```bash
TARGETS="matmul:triton:large matmul:pytorch:large layer_norm:tilelang:large \
         argmax:tilelang:large conv2d:triton:large" \
    bash experiments/ncu_counters.sh
```

## (f) Expected differences to anticipate (and how to read them)

These are *predictions* — the point of the replay is to confirm/refute them:

- **Memory bandwidth:** A100 HBM2e ≈ 1.5–2.0 TB/s, H100 HBM3 ≈ 3.0–3.3 TB/s vs
  Ada GDDR6 **360 GB/s** (4–9×). **Bandwidth-bound kernels** (the reductions,
  LayerNorm, elementwise) get much faster in absolute ms; the *DSL-vs-library
  ratio* is what to compare across arches, not raw latency.
- **L2 cache:** A100 = 40 MB, H100 = 50 MB vs Ada 48 MB. Combined with far higher
  DRAM BW, **RC2b (L2/BW-bound matmul at 16384²) may shrink or shift** — the
  1.5 GB working set still ≫ L2, but re-streaming costs less. Watch
  `lts__t_sector_hit_rate.pct` and the DRAM-bytes ÷ 1.5 GB re-stream factor.
- **GEMM hardware:** H100 adds **TMA + wgmma** async pipelines that cuBLAS
  exploits heavily; **the GEMM gap (cuBLAS vs DSL) may widen on H100** if the DSLs
  don't emit wgmma, or shrink if their autotuners adapt. A100 (Ampere
  tensor cores, no TMA/wgmma) is the intermediate point.
- **RC0a TileLang LayerNorm anomaly (the central result):** the 314× collapse
  and the `T.serial`→`T.reduce` recovery are a **memory-latency exposure** —
  Ada NCU shows `warp_stall_long_scoreboard` ≈ 105 cycles with
  `warp_stall_barrier` ≈ 0, so the dominant cost is serialized memory-latency
  hiding, **not** barrier synchronization (the submitted-PDF framing). It
  should reproduce on **both** A100 and H100 — `long_scoreboard` dominating
  (set C) and `…data_bytes_per_sector…pct` < 100% (set A) **independent of
  arch**. That cross-arch consistency is itself a strong result for R1-Q5/R3.
- **Registers/spill (RC3 — TileLang LayerNorm, NOT conv):** Ada NCU shows
  Triton conv `n_spills = 0` for filters 1×1–7×7. The 51.5 GB spill / 254
  regs/thread / 16.5% occupancy lives in **TileLang LayerNorm**, not in conv.
  On A100/H100 (same 65,536 regs/SM as Ada) the TileLang norm spill is
  expected to reproduce; the conv per-filter `n_regs` trend (set B) is the
  portable claim measured by `exp_conv_filters.py`, and the spill column for
  conv is expected to stay 0 on the new arches as well.

**Framing for the camera-ready (R1-Q5, R3):** *"We replay the suite on A100 and
H100 with an unmodified portable harness. The RC0 LayerNorm anomaly reproduces
arch-independently (consistent with its sync/scalar-load mechanism); the
L2/bandwidth-bound root cause (RC2b) and the GEMM gap shift with HBM bandwidth
and H100's TMA/wgmma, which we report and attribute. This tests whether each root
cause is a property of the DSL or of the hardware."*

---

### Artifacts produced per arch
```
experiments/results/<gpu_slug>/
    run_all.log                       # full suite log (tee'd)
    fp32_gemm.csv                     # Exp 2: TF32 root cause (W1/W11)
    correctness_edge.csv              # edge-case numerical correctness
    conv_filters_small.csv            # Exp 3a: 1/3/5/7 + depthwise, Ada-safe shape
    conv_filters_large.csv            # Exp 3a: 1/3/5/7 + depthwise, paper Table-2 shape
    conv_mitigation_small.csv         # Exp 3b: AKO4ALL conv2d_triton on small shape
    conv_mitigation_large.csv         # Exp 3b: AKO4ALL conv2d_triton on large shape
    fused_baselines.csv               # Eager vs torch.compile vs DSL
    winograd_isolation.csv            # Exp 4: Winograd eligible vs not (RC4)
    cudnn_winograd_3x3.log            # cuDNN algorithm-selection log for 3x3
    autotune_matmul.csv               # §5 vs §7.3 matmul autotune reconcile
    significance.csv                  # Step (d.5): locked-clock dispersion (manual)
    significance_smoke.csv            # written by --smoke runs (plumbing check)
    clock_lock.txt                    # locked-clock verification (lock_clocks.sh / run_pipeline.sh)
    mitigation_retime.csv             # retime_mitigation.py: all mitigations (AKO4ALL norm/argmax/matmul
                                      #   + optimized reduction/softmax family from opt_kernels/)
    ncu_summary.csv                   # rolled-up RC table for tab:rootcauses
    NCU_FINDINGS.md                   # written by consolidate_ncu.py
    ncu/
        <kernel>_<impl>_large_loads.csv    # set A  (vectorized loads)
        <kernel>_<impl>_large_regs.csv     # set B  (regs + spill + occupancy)
        <kernel>_<impl>_large_stalls.csv   # set C  (warp-stall breakdown)
        <kernel>_<impl>_large_l2dram.csv   # set D  (L2 / DRAM)
```
Plus, written outside the per-arch dir (by `regen_profile.sh`):
```
ViperBench/results/profile.<short>.csv         # untuned pytorch/triton/tilelang (this arch)
ViperBench/results/profile.<short>.tuned.csv   # + triton_tuned/tilelang_tuned (with --tuned)
ViperBench/results/tuning_cache.json           # gains this arch's best-config keys from the sweep
```
The optimized reduction/softmax kernels (`experiments/opt_kernels/*_opt.py`) are
arch-portable source; `retime_mitigation.py` re-times them per arch.

Each CSV is self-describing (the harness prepends `gpu_name`, `sm`,
`timestamp_utc`), so the A100/H100 results are unambiguous when collated with
Ada's into the cross-architecture comparison tables.

> **Future work:** a small `experiments/compare_archs.py` that diffs
> `ncu_summary.csv`, `conv_filters_*.csv`, `autotune_matmul.csv`,
> `fp32_gemm.csv`, `winograd_isolation.csv` across multiple `<gpu_slug>` result
> dirs would automate the cross-arch table assembly the paper's revision needs.
> Until then, the per-arch CSVs are pre-aligned in schema, so a single `pandas`
> read+merge on `(kernel, impl, shape)` is sufficient.
