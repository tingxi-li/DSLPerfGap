# Acceptance-Critical Experiment Protocols — ASE 2026 #4134 Rebuttal

Companion to `REBUTTAL_GAME_PLAN.md` (strategy + rebuttal draft) and `REBUTTAL_EXPERIMENT_PROTOCOLS.md` (cheap "do-now" wins). This file covers the experiments that substantiate the paper's **central contribution** — a hardware-counter-grounded root-cause taxonomy — and answer the reviewers' hardest questions. Reviewer key: **R1 = 4134A, R2 = 4134B, R3 = 4134C.**

## Environment facts verified on the paper's GPU (read first — they set every status below)

| Fact | Verified value | Consequence |
|---|---|---|
| GPU | 2× **RTX 4000 Ada**, sm_89, AD104, 20 GB, L2 = 48 MB, BW = 360 GB/s | paper's exact HW; `ncu` supports `ad104` |
| `ncu` | `/usr/local/cuda/bin/ncu` v2024.3.2.0 — works (`--list-sections` OK) | tool fine |
| **Counter permission** | `/proc/driver/nvidia/params` → `RmProfilingAdminOnly: 1`; **every** ncu collection returns `ERR_NVGPUCTRPERM`, including `launch__registers_per_thread` and default sections | **BLOCKER** — no ncu counter collects until an admin sets `NVreg_RestrictProfilingToAdminUsers=0` + reloads the driver, or ncu runs as root. `sudo` is disallowed by project rules. |
| Admin-free register/spill fallback | Triton 3.4.0 exposes `kernel.n_regs` / `kernel.n_spills`; `ptxas -v`; `cudaFuncAttributes.numRegs` — all work with no perms | RC3 / W13 register & spill data obtainable **today**; warp-stall / vectorized-load / L2 counters are **not** |
| Clocks | persistence Disabled; max SM 3105 MHz | W8 clock locking also needs admin (`nvidia-smi -pm/-lgc`) |

> **Single most important consequence:** the headline counter table (W3 / R1-Q4) is gated on **one** sysadmin action (a 2-line modprobe change + reboot, ~15 min). **Request it on Day 0** — it is the cheapest, highest-leverage action available. If granted in-window → Experiments 1/3B/5 become *done*. If not → they are *committed*, but RC3 (registers) and W13 are still substantiable now via the Triton/ptxas path.

---

## Experiment 1 — Nsight hardware-counter collection (HIGHEST IMPACT)
**Substantiates:** the counter-grounded taxonomy (§3.3/§3.4); today the artifact has **zero** measured counters. Produces exactly the three families R1-Q4 asked for. **Reviewer Q:** W3 / R1-Q4 / R2-Q4.

### Step 0 — UNBLOCK counters (admin-only, ~15 min). Hand the sysadmin:
```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo dracut -f                # RHEL9 here (use update-initramfs -u on Debian/Ubuntu)
sudo reboot                   # or rmmod/modprobe nvidia* if no GUI attached
# verify after reboot (no sudo): grep RmProfilingAdminOnly /proc/driver/nvidia/params  → must read 0
```
Fallback if admin declines: a one-off `sudo ncu ...`, else use Experiment 3 Path A for the register subset.

### Step 1 — single-launch harness `ViperBench/profiling/run_one_kernel.py`
`ncu` must profile exactly ONE kernel invocation or it serializes every launch (hours). Warm up first (triggers TileLang/Triton JIT + autotune so compile is not profiled), then run the target once under an NVTX range:
```python
# python run_one_kernel.py <kernel> <impl> <size>
import sys, os, importlib.util, torch, torch.cuda.nvtx as nvtx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from benchmark import get_test_cases                 # reuse the exact paper shapes
kernel, impl, size = sys.argv[1:4]
kdir = os.path.join(os.path.dirname(__file__), '..', kernel)
def load(m):
    s = importlib.util.spec_from_file_location(m, os.path.join(kdir, m + '.py'))
    mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod); return mod
mod = load(f'{impl}_impl')
_, fn_name, args, kwargs, desc, _ = next(c for c in get_test_cases()[kernel] if c[0] == size)
fn = getattr(mod, fn_name); kwargs = kwargs or {}
for _ in range(5): fn(*args, **kwargs)               # JIT + autotune warmup (NOT profiled)
torch.cuda.synchronize()
nvtx.range_push('TARGET'); fn(*args, **kwargs); torch.cuda.synchronize(); nvtx.range_pop()
```
Discover the kernel-name regex / launch index with a cheap pass, then profile only that launch. `--target-processes all` is **required** (Python spawns the CUDA process).

### Step 2 — the four metric sets (verified plausible for sm_89 / ncu 2024.3.2)

**(A) Vectorized / global-load efficiency — conv vs. GEMM (R1-Q4 #1):**
```bash
ncu --target-processes all --nvtx --nvtx-include "TARGET/" --launch-count 1 --metrics \
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
sm__sass_inst_executed_op_global_ld.sum \
--csv --log-file conv_loads.csv python ViperBench/profiling/run_one_kernel.py conv2d triton large
```
`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` = **vectorized-load fraction** (100% = full 32 B/sector ≈ LDG.128-class; lower = scalar/uncoalesced). Run the same set on `matmul triton large` and **critically** `layer_norm tilelang large` — proves/refutes RC0b's "scalar 16-bit loads" claim.

**(B) Register usage + spill proxy (R1-Q4 #2, RC3):**
```bash
ncu --target-processes all --nvtx --nvtx-include "TARGET/" --launch-count 1 --metrics \
launch__registers_per_thread,launch__occupancy_limit_registers,\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
--csv --log-file regs_matmul.csv python ViperBench/profiling/run_one_kernel.py matmul triton large
```
Non-zero `local_op_ld/st` bytes = **register spill to local memory**; `sm__warps_active...pct` = achieved occupancy.

**(C) Warp-stall breakdown (R1-Q4 #3, RC0):**
```bash
ncu --target-processes all --nvtx --nvtx-include "TARGET/" --launch-count 1 --section WarpStateStats --metrics \
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio \
--csv --log-file stalls_argmax.csv python ViperBench/profiling/run_one_kernel.py argmax tilelang large
```
RC0's story is "24 `__syncthreads()` per fragment" → expect **`stalled_barrier`** to dominate for the TileLang `T.serial` kernels (argmax, max_reduction, the `T.serial` layer_norm baseline at `ViperBench/layer_norm/tilelang_impl.py:44-45,50-51`). If `long_scoreboard` dominates instead → memory-latency-bound, not sync-bound — the measurement decides.

**(D) L2 / DRAM (RC2b — feeds Experiment 5):**
```bash
ncu --target-processes all --nvtx --nvtx-include "TARGET/" --launch-count 1 --metrics \
lts__t_sector_hit_rate.pct,dram__bytes_read.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
--csv --log-file l2_matmul.csv python ViperBench/profiling/run_one_kernel.py matmul triton large
```

### Targets (minimal high-value set, ~8 kernel×impl pairs)
Triton conv2d (large); Triton matmul + cuBLAS-via-`pytorch_impl` matmul (large); TileLang `T.serial` layer_norm (large); TileLang argmax (large); TileLang max_reduction (large). Profile cuBLAS so the GEMM column has the reference the paper compares against.

### Output table template (fill when run) → consolidate to `ViperBench/results/ncu_counters.csv`
| RC | Kernel (impl) | Counter | Measured | Interpretation |
|---|---|---|---|---|
| RC0a | argmax (TileLang) | `…stalled_barrier…` | ⟨⟩ | sync-bound if ≫ others |
| RC0b | layer_norm (TileLang `T.serial`) | `…data_bytes_per_sector…pct` | ⟨⟩ | <100% ⇒ scalar loads |
| ref | matmul (cuBLAS) | same | ⟨⟩ | ≈100% expected |
| RC3 | matmul/conv (Triton) | `registers_per_thread`, `local_op_st` | ⟨⟩ | spills if local_st>0 |
| RC2b | matmul (Triton) | `lts__t_sector_hit_rate.pct` | ⟨⟩ | low ⇒ L2 thrash |

**Runtime:** harness+discovery ≈ 1 h; each isolated single-launch run ≈ 20–90 s (**never** use `--set full`); ~8 targets × 4 sets ≈ **2–3 h** once unblocked.
**Status:** **UNDERWAY → done IF admin enables counters; else COMMITTED.** The register half (set B) is obtainable now via Exp 3 Path A.

---

## Experiment 2 — FP32 GEMM correctness root-cause (TF32 truncation hypothesis)
**Substantiates:** converts the unattributed FP32 failure (99.6% mismatch, 2067× rel err @ 4096×2048×1024, currently only a copied GitHub issue in `AKO4ALL/context/known_github_issues.md:71-160`) into a **definitive, reproducible** root cause + a permanent regression test. Prime suspect: `T.gemm` silently uses the **TF32** path (10-bit mantissa) for `dtype="float32"` — supported by `ViperBench/batched_matmul/tilelang_impl.py:10` ("T.gemm uses TF32"). **Reviewer Q:** W1 / W11 / R1-Q1. **No admin; runs now.**

### Protocol — new file `ViperBench/matmul/test_fp32.py`, M,N,K = 4096,2048,1024, fp32
Compare four arms against `X @ W` (PyTorch fp32, TF32 disabled = reference):
- **Arm A — `T.gemm` fp32:** exact kernel from `known_github_issues.md:79-113`. **Expected: reproduces 99.6% / ~2067×.**
- **Arm B — non-`T.gemm` fp32 accumulation:** manual MAC pattern from `batched_matmul/tilelang_impl.py:38-58` adapted to 2D. **Expected: passes @ 1e-5** → isolates cause to `T.gemm`, not layout/indexing.
- **Arm C — TF32 reference:** does Arm A match a `torch.backends.cuda.matmul.allow_tf32=True` reference within ~1e-2 rel but NOT the fp32 reference? If yes → **definitively TF32 mantissa truncation**, not a bug.
- **Arm D:** retry Arm A with any TileLang fp32/disable-TF32 pass-config (grep installed `tilelang` for `tf32`/`precision`); if none exists, *that is the finding* — no user knob, so fp32 `T.gemm` is unavoidably TF32.

**Magnitude sanity check (cheap, supports conclusion):** TF32 truncates mantissa 23→10 bits ⇒ per-element rel err ~2⁻¹¹≈5e-4; over K=1024 with cancellation near zero-valued outputs, isolated 10²–10³× *relative* errors at *near-zero* entries are the expected TF32 signature (the 2067× is at index (2778,409), a near-zero output). A layout bug instead gives ~100% large *absolute* error everywhere — the distinguishing test.

**Deliverables:** (1) one-sentence root cause (expected: *"not a bug — `T.gemm` lowers fp32 to the TF32 tensor-core path; the 2067× rel err occurs only at near-zero outputs via cancellation; Arms B and C confirm"*); (2) `test_fp32.py` that reproduces the failure (Arm A asserted to fail fp32 tol but pass TF32 tol).
**Runtime:** ~30–45 min. **Status: DONE by rebuttal** (local, deterministic).

---

## Experiment 3 — RC3: register pressure / occupancy across 3×3 / 5×5 / 7×7 conv
**Substantiates:** RC3 is currently *hypothetical* (5×5/7×7 never perf-benchmarked; harness hardcodes 3×3). Gives measured register/occupancy degradation vs. filter size; also resolves **W13** with measured **Ada** numbers. **Reviewer Q:** W3 / R1-Q2 / R2-Q2.

**Configs:** fix input `(32,256,128,128)` fp16, padding to preserve spatial; weight `(256,256,k,k)`, **k ∈ {1,3,5,7}** (matches the paper's conv-large shape). Triton kernel loops `for h in range(kernel_height): for w in range(kernel_width)` (`conv2d/triton_impl.py:54-55`) → `accum` register lifetime grows with k².

**Path A — admin-free register/spill (RUNS NOW):** Triton 3.4.0 returns the compiled handle:
```python
h = conv2d_forward_kernel[grid](...)        # CompiledKernel
print(k, h.n_regs, h.n_spills, h.metadata.num_warps)
```
Tabulate `n_regs`/`n_spills` for k=1,3,5,7. **Expected:** `n_regs` rises with k; `n_spills`>0 first at 5×5 or 7×7 — substantiates RC3 **without admin**. (Cross-check: `ptxas -v`, or `cudaFuncAttributes.numRegs`.)

**Path B — ncu occupancy (needs Step-0 unblock):** reuse Exp 1 metric set (B) per k (add `large_k5`/`large_k7` to `get_test_cases()`).

| Filter | n_regs | n_spills | regs/thr (ncu) | occ % | local-store B | latency ms |
|---|---|---|---|---|---|---|
| 1×1 / 3×3 / 5×5 / 7×7 | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |

**Runtime:** Path A ≈ **30 min now**; Path B ≈ +45 min after unblock. **Status: Path A DONE by rebuttal (RC3 evidence + W13 Ada numbers); Path B UNDERWAY/COMMITTED.** Doubles as the **W6** conv-filter latency data reviewers asked for.

---

## Experiment 4 — RC4: Winograd isolation in the conv gap
**Substantiates:** RC4 has no isolation experiment today. **Reviewer Q:** W3 / R1 (detailed comment). **Runs now (partial); no admin.**

**Controllable from PyTorch:** `cudnn.benchmark` (on/off changes algo selection), `cudnn.allow_tf32`, `cudnn.deterministic=True` (forbids most Winograd/non-deterministic algos — closest "disable Winograd" lever), `cudnn.enabled=False` (drops to non-cuDNN path). **Not controllable:** no public PyTorch API to force/forbid a specific cuDNN algorithm by name — a clean Winograd on/off A/B needs the raw cuDNN C API (revision scope). State this plainly.

**Three complementary measurements (cuDNN baseline, conv-large shape):**
1. **Algo-selection A/B:** `F.conv2d` 3×3 with `deterministic=False` vs `deterministic=True`; the latency delta upper-bounds Winograd's 3×3 contribution.
2. **Winograd-eligible vs. ineligible (cleanest proxy):** Winograd F(2,3)/F(4,3) applies to **3×3 stride-1** but **not** 1×1, large filters, or **stride-2**. Time cuDNN & Triton/TileLang at 3×3 s1 (eligible) vs 3×3 s2 and 5×5/7×7 (ineligible; reuse Exp 3 shapes). If the DSL-vs-cuDNN gap **shrinks markedly** when Winograd is unavailable → residual 3×3 gap attributable to Winograd. No special API; runs today.
3. **Confirm cuDNN's choice (observability):** run under `CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=stdout` and grep for `winograd` in the engine name — turns the assumption into an observation. (Also yields **N1** evidence about what cuDNN actually does.)

| Config | cuDNN ms | Triton ms | TileLang ms | gap | Winograd? |
|---|---|---|---|---|---|
| 3×3 s1 / 3×3 s1 determ. / 3×3 s2 / 5×5 / 7×7 | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |

**Runtime:** ~45–60 min. **Status: DONE by rebuttal** (eligible-vs-ineligible proxy + cuDNN-log confirmation); **COMMITTED** for a raw-cuDNN-API hard Winograd-off run.

---

## Experiment 5 — RC2b: L2 cache residency at 16384² matmul
**Substantiates:** RC2b ("DSL matmul thrashes the 48 MB L2; ~1.5 GB working set ≫ L2"), asserted from capacity arithmetic with no measured hit rate. **Reviewer Q:** W3. **Needs Step-0 unblock.**

**Anchor arithmetic (state regardless):** 16384² fp16: A+B+C = 3·16384²·2 B = **1.5 GB** ≫ 48 MB L2 (32×). End-to-end governed by L2 *reuse*, not capacity.

**Protocol:** Exp 1 metric set (D) on **cuBLAS / Triton / TileLang** at the 16384² shape. **Decisive cross-check:** arithmetic intensity = 2·16384³ FLOP ÷ `dram__bytes_read.sum`; cuBLAS should show higher L2 hit rate and far fewer DRAM bytes than the DSLs (DSL DRAM bytes ≫ the 1.5 GB minimum = quantified L2 thrash).

| Impl | L2 hit % | DRAM bytes | DRAM BW % | re-stream factor (÷1.5 GB) | latency ms (paper) |
|---|---|---|---|---|---|
| cuBLAS / Triton / TileLang | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | 114.7 / 361.9 / 203.3 |

**Runtime:** ~30 min after unblock. **Status: UNDERWAY → done IF unblocked; else COMMITTED** (arithmetic + latencies presentable now).

---

## Experiment 6 — Second-GPU / cross-architecture (the hard one)
**Substantiates:** external validity (every reviewer; §10 already promises it). **Reviewer Q:** W5 / R1-Q5 / R3. **CANNOT run locally** (only RTX 4000 Ada; no A100/H100) → the principal revision commitment.

**Why it must be run, not reasoned:** A100/H100 differ exactly where the taxonomy lives — HBM2e/HBM3 at 1.5–3.3 TB/s vs Ada's 360 GB/s GDDR6, larger L2, and (H100) TMA + wgmma async-pipeline hardware cuBLAS exploits. RC2b (L2/BW-bound) and the GEMM gap could shrink or invert.

**Minimal high-value subset (~1 GPU-day/arch):** (1) headline category gaps — matmul 16384², conv2d (32,256,128,128) 3×3, layer_norm 8192² (RC0 anomaly), argmax/max_reduction (8192,32768); (2) does the 314× LayerNorm collapse + `T.serial`→`T.reduce` recovery reproduce? (likely yes — it's an arch-independent authoring/sync bug, and that consistency is itself a strong result); (3) Exp 4's Winograd proxy; (4) Exp 1 sets A+C counter spot-check (identical metric IDs; ncu is arch-portable).

**Access & effort:** one cloud A100 80 GB + one H100 80 GB (AWS p4d/p5, GCP a2/a3, Lambda, RunPod) or an institutional slot — **request profiling rights / root** (same `NVreg_RestrictProfilingToAdminUsers` blocker applies). Portable harness (`pip install tilelang triton torch`; JIT recompiles for the target SM). ≈ **2–3 GPU-days** total.

**Framing:** *"Cross-architecture validation is the single experiment we cannot complete in the rebuttal window; we commit to A100 and H100 results in the camera-ready, and have a portable harness + cloud access to do so."* Pair with what you can argue now: the RC0 LayerNorm anomaly is architecture-independent **by construction** (sync barriers + scalar loads), so its generalization is argued from mechanism even before the numbers land. **Status: COMMITTED-to-revision.**

---

## Recommended execution order & time budget

**Action 0 (Day 0, parallel with all):** email the admin the `NVreg_RestrictProfilingToAdminUsers=0` change + reboot. This one request unblocks Experiments 1, 3B, 5.

| # | Experiment | Admin? | Effort | Status target |
|---|---|---|---|---|
| 1 | **Exp 2 — FP32 root-cause** | No | ~45 min | **DONE** (closes W1/W11) |
| 2 | **Exp 3 Path A — register/spill table** (+ W6 conv latencies) | No | ~30 min | **DONE** (RC3 + W13, no admin) |
| 3 | **Exp 4 — Winograd eligible/ineligible proxy + cuDNN log** | No | ~1 h | **DONE** (+ N1 evidence) |
| 4 | **Exp 1 harness + regex discovery** | build | ~1 h | prerequisite for 5–6 |
| 5 | **Exp 1 — counter collection** (8 targets × 4 sets) | **Yes** | ~2–3 h | **DONE if unblocked, else UNDERWAY** |
| 6 | **Exp 5 — L2 residency** + **Exp 3 Path B occupancy** | **Yes** | ~1 h | rides on #5 |
| 7 | **Exp 6 — cross-arch** | external GPU | 2–3 GPU-days | **COMMITTED** |

**Completable now with no admin (~3.5 h):** Exp 2, 3A, 4 + the Exp-1 harness — converts the FP32 failure (W1), RC3 register evidence (W3/W13), and Winograd isolation (RC4) from *asserted* to *measured*. **If admin enables counters (~15 min of their time):** add Exp 1, 5, 3B (~4–5 h) for the full counter table R1-Q4 demands — the highest-impact deliverable. **Pure revision commitment:** Exp 6 (second GPU) only.
