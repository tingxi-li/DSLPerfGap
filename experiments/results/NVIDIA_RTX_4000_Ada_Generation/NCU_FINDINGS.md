# Hardware-counter findings — RTX 4000 Ada (sm_89) — ASE 2026 #4134

*Collected with `ncu_counters.sh` after the profiling permission was unblocked (`RmProfilingAdminOnly: 0`, box rebooted 2026-05-25). 24 collections × 6 kernels × 4 metric families, **0 failed**. Each kernel profiled as a single NVTX-`TARGET`-isolated launch; TileLang's compute kernel pinned via `--kernel-name regex:func_kernel` (its unified-API wrapper also launches PyTorch preamble copies). Raw CSVs in `ncu/`, tidy form in `ncu_summary.csv`. Portable: identical collection on A100/H100.*

This is the **counter-grounded evidence the paper's taxonomy was missing** (the artifact previously had zero measured counters — reviewer R1-Q4 / W3). It both **grounds** the taxonomy and **corrects two micro-mechanisms**.

## The table

| metric (RC) | matmul cuBLAS | matmul Triton (plain) | conv2d Triton | layer_norm TileLang | argmax TileLang | max_reduction TileLang |
|---|--:|--:|--:|--:|--:|--:|
| kernel | `ampere…s16816gemm` | `matmul_kernel` | `conv2d_forward_kernel` | `func_kernel` | `func_kernel` | `func_kernel` |
| **RC3** regs/thread | 218 | 154 | 128 | **254** | 39 | 39 |
| **RC3** achieved occ % | 16.7 | 25.0 | 33.3 | **16.5** | 80.7 | 64.4 |
| **RC3** spill load B | 0 | 0 | 0 | **51.5 GB** | 0 | 0 |
| **RC3** spill store B | 0 | 0 | 0 | **34.4 GB** | 0 | 0 |
| **RC0** stall long_scoreboard | 0.06 | 22.2 | 10.2 | **104.9** | 27.4 | 86.7 |
| **RC0** stall barrier (sync) | 0.09 | 8.02 | 1.54 | **0** | **0** | **0** |
| **RC1** global-load eff % (bytes/sector) | 99.7 | 0\* | **36.4** | 50.0 | 12.5 | 12.5 |
| **RC2b** L2 hit % | 91.0 | 66.2 | 95.2 | 53.1 | **0.64** | 38.6 |
| **RC2b** DRAM throughput % | 27.2 | 56.7 | 19.5 | **90.2** | 10.2 | 20.8 |
| kernel time (ms) | 123.6 | 363.1 | 36.6 | 271.2 | 30.9 | 34.1 |

\* Triton matmul reads 0% global-load efficiency because it stages A/B tiles via `cp.async` (LDGSTS), which bypasses the regular-LDG counter — not an uncoalesced-load signal. Use cuBLAS (99.7%) vs conv (36.4%) for the RC1 coalescing comparison, both of which use ordinary `LDG`.

## What the counters say

**1. cuBLAS is the efficient reference, and it is efficient for measurable reasons.** 99.7% load efficiency, essentially zero stalls (long_scoreboard 0.06, barrier 0.09), 91% L2 hit. The DSL kernels each miss this in a *different, now-measured* way.

**2. RC0 — the stall mechanism is memory latency, NOT synchronization.** The protocol set up an explicit test: *does `stalled_barrier` dominate the `T.serial` TileLang kernels (sync-bound) or does `long_scoreboard` (memory-latency-bound)?* The answer is unambiguous: **all three TileLang reductions have `barrier ≈ 0` and `long_scoreboard` 27–105.** They are **memory-latency-bound, not synchronization-bound.** If the paper attributes the `T.serial` slowdown to `__syncthreads`/barrier stalls, that wording must be corrected — `T.serial` lowers to a *per-thread serial accumulation* (no cross-thread barriers), whose cost is serialized memory latency (and, for layer_norm, spill traffic).

**3. RC3 — register spilling is real, but kernel-specific.** This refines the earlier (Triton-conv-only) finding:
   - **TileLang layer_norm spills catastrophically:** 254 regs/thread *and* **51.5 GB local-load + 34.4 GB local-store** in a single launch, occupancy pinned to 16.5%. This is a genuine register-pressure/spill root cause — and it sits exactly on the paper's largest TileLang normalization anomaly.
   - **Triton conv does NOT spill:** 0 local bytes at every filter size (matches `n_spills=0` from kernel attributes). The conv filter-size gap is occupancy (33%) + coalescing (36% load eff), not spilling.
   - So "register spilling" is the mechanism for the **TileLang normalization anomaly**, *not* for the **conv-filter gap**. State each precisely; do not generalize either way.

**4. This explains why the `T.serial → T.reduce` mitigation gives ~1224×.** The counters show the original `T.serial` layer_norm is 254 regs + 86 GB spill traffic + long-scoreboard latency. `T.reduce` removes the per-thread accumulation array → no spill → the kernel stops being latency/spill-bound. The mitigation's mechanism is now measured, not asserted.

**5. RC1 (coalescing) and RC2b (memory) are grounded.** Conv load efficiency 36.4% vs cuBLAS 99.7% (uncoalesced/strided conv access). argmax L2 hit 0.64% with 10% DRAM throughput → latency-bound streaming with no reuse. layer_norm DRAM 90.2% → bandwidth-saturated (by spill traffic).

## How this lands in the rebuttal (honest framing)

- **Positive:** "We have collected the hardware counters underpinning the taxonomy (vectorized-load efficiency, register/occupancy, warp-stall breakdown, L2/DRAM) for the six representative kernels; full data in the revision." — true, done.
- **Corrections to fold into the revision (and not contradict in the rebuttal):**
  - The TileLang reduction bottleneck is **long-scoreboard (memory-latency)**, not **barrier (sync)** stalls.
  - **Register spilling** is confirmed for the **TileLang normalization** kernel (51 GB), and **absent** for the **Triton conv** kernels — attribute it to the right kernel.
- **Do not** claim "barrier stalls dominate" or "conv spills registers" — the counters say otherwise. The measured story (latency-bound + kernel-specific spilling, explaining the T.reduce fix) is stronger and is what the reviewers asked us to produce.
