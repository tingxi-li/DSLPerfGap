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
pip install torch triton tilelang
```
- **TileLang and Triton JIT-recompile for the target SM automatically.** First
  call on A100 (sm_80) / H100 (sm_90) triggers a fresh compile + autotune; the
  `run_one_kernel.py` warmup loop and `_harness.time_kernel` warmup absorb that
  compile so it is never timed or profiled.
- Match the CUDA toolkit to the driver. For **H100 (sm_90)** you need a
  CUDA 12.x toolchain and an `ncu` ≥ 2023.x that knows `gh100`; CUDA 12.4+ /
  Nsight Compute 2024.x (what we used on Ada) covers both A100 and H100.
- Sanity-check the portability layer (prints the *detected* GPU, no hardcoding):
  ```bash
  python experiments/_harness.py     # banner + device_info + a trivial matmul timing
  ```

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

# 2. Hardware counters (after permissions granted):
bash experiments/ncu_counters.sh
#    -> experiments/results/<A100_or_H100_slug>/ncu/*.csv
```
- The GPU slug (e.g. `NVIDIA_A100-SXM4-80GB`, `NVIDIA_H100_80GB_HBM3`) is
  derived from `torch.cuda.get_device_name`, so **A100/H100 will not collide
  with Ada** (`NVIDIA_RTX_4000_Ada_Generation`). Re-running overwrites only that
  arch's CSVs (idempotent).
- If the box has multiple GPUs, pin one: `CUDA_VISIBLE_DEVICES=1 bash experiments/run_all.sh`.
- Counter metric IDs in `ncu_counters.sh` are **arch-portable** — the same
  strings work on sm_80/sm_90; no edits needed.

## (e) Minimal high-value subset (when GPU time is limited, ~1 GPU-day/arch)

If you can't run the full suite, this subset still answers the reviewers'
generalization question:

| Target | Why it matters | How |
|---|---|---|
| **matmul 16384²** | headline GEMM gap; RC2b L2 thrash; TMA/wgmma effect on H100 | `run_one_kernel.py matmul {triton,pytorch,tilelang} large` + counter set D |
| **conv2d 3×3** `(32,256,128,128)` | headline conv gap; Winograd eligibility | `run_one_kernel.py conv2d triton large` + sets A/B; `exp_winograd_isolation.py` |
| **layer_norm 8192²** | the **RC0 anomaly** (sync/scalar-load bug) — expected to reproduce arch-independently | `run_one_kernel.py layer_norm tilelang large` + sets A/C |
| **argmax + max_reduction** `(8192,32768)` | RC0a `stalled_barrier` (24 `__syncthreads`/fragment) | `run_one_kernel.py {argmax,max_reduction} tilelang large` + set C |
| **counter spot-check** | sets **A** (vectorized loads) + **C** (warp stalls) on the five above | `ncu_counters.sh` (override `TARGETS=...` to trim) |

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
- **RC0 TileLang LayerNorm anomaly (the central result):** the 314× collapse and
  the `T.serial`→`T.reduce` recovery stem from **excess sync barriers + scalar
  loads in the kernel source**, not from any HW feature. It should reproduce on
  **both** A100 and H100 — `stalled_barrier` dominating (set C) and
  `…data_bytes_per_sector…pct` < 100% (set A) **independent of arch**. That
  cross-arch consistency is itself a strong result for R1-Q5/R3.
- **Registers/spill (RC3):** `launch__registers_per_thread` and occupancy limits
  differ per SM (A100/H100 have 65536 regs/SM like Ada but different occupancy
  curves); the *trend* (regs rising with conv filter size 1→3→5→7, spills
  appearing at 5×5/7×7) is the portable claim, re-measured by
  `exp_conv_filters.py` + set B.

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
    run_all.log                     # full suite log (tee'd)
    exp_fp32_gemm.csv               # Exp 2  (TF32 root cause)
    exp_correctness_edge.csv        # edge-case correctness
    exp_conv_filters.csv            # Exp 3  (regs/latency vs 1/3/5/7; RC3/W13/W6)
    exp_fused_baselines.csv
    exp_winograd_isolation.csv      # Exp 4  (Winograd eligible vs ineligible; RC4)
    exp_autotune_matmul.csv
    ncu/
        <kernel>_<impl>_large_loads.csv    # set A  (vectorized loads)
        <kernel>_<impl>_large_regs.csv     # set B  (regs + spill + occupancy)
        <kernel>_<impl>_large_stalls.csv   # set C  (warp-stall breakdown)
        <kernel>_<impl>_large_l2dram.csv   # set D  (L2 / DRAM)
```
Each CSV is self-describing (the harness prepends `gpu_name`, `sm`,
`timestamp_utc`), so the A100/H100 results are unambiguous when collated with
Ada's into the cross-architecture comparison tables.
