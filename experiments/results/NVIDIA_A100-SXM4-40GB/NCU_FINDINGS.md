# Hardware-counter findings — A100-SXM4-40GB (sm_80, Ampere)

Canonical RC interpretation of the A100-SXM4 Nsight Compute sweep (`ncu_counters.sh` → `ncu/` →
`consolidate_ncu.py` → `ncu_summary.csv`). Counters are the same arch-portable sets used on the
RTX 4000 Ada (sm_89) and GH200 (sm_90) runs, so the taxonomy is directly comparable across
architectures. Each kernel is profiled at the paper "large" shape, exactly one launch inside an
NVTX `TARGET` range. Stack: torch 2.8.0+cu126 / triton 3.4.0 / tilelang 0.1.6.post1, ncu 2024.3.2,
driver 580.105.08, clocks locked 1215/1215 (sustained).

## The table

| metric (RC) | matmul cuBLAS | matmul Triton | conv2d Triton | layer_norm TL | argmax TL | max_red TL | logsumexp TL_opt |
|---|--:|--:|--:|--:|--:|--:|--:|
| kernel | `cutlass_80_tensorop_f16_s16816gemm` | `matmul_kernel` | `conv2d_forward_kernel` | `func_kernel` | `func_kernel` | `func_kernel` | `func_kernel` |
| **RC3** regs/thread | 218 | 166 | 214 | **254** | 32 | 36 | **93** |
| **RC3** achieved occ % | 12.5 | 18.7 | 12.5 | **12.4** | 27.9 | 73.3 | **30.5** |
| **RC3** spill load B | 0 | 0 | 0 | **51.5 GB** | 0 | 0 | **0** |
| **RC3** spill store B | 0 | 0 | 0 | **34.4 GB** | 0 | 0 | **0** |
| **RC0** stall long_scoreboard | 0.01 | 1.49 | 3.40 | **58.73** | 41.46 | 38.88 | 12.50 |
| **RC0** stall barrier (sync) | 0.01 | 0.87 | 0.41 | **0** | **0** | 4.96 | 1.06 |
| **RC1** global-load eff % (bytes/sector) | 0\* | 0\* | 26.3 | 50.0 | 12.5 | **100** | **100** |
| **RC2b** L2 hit % | 74.2 | 61.6 | 94.6 | 54.7 | 49.8 | 22.8 | 23.1 |
| **RC2b** DRAM throughput % | 40.1 | 80.8 | 6.5 | 56.6 | 7.0 | **94.3** | **93.3** |
| kernel time (ms) | 37.43 | 56.63 | 13.28 | 99.10 | 9.87 | **0.37** | **0.37** |

\* the cuBLAS `cutlass_80…s16816gemm` and the Triton GEMM use `cp.async`/`ldmatrix` bulk loads on
Ampere, so the classic `global_op_ld` bytes/sector counter reads 0 — not a vectorization failure.

## What the counters say

**RC0 (corrected: memory-latency, not barrier sync) reproduces on Ampere.** The slow TileLang
norm/reduction kernels stall on `long_scoreboard` (memory latency), with `barrier`=**0**: layer_norm
58.73 / 0, argmax 41.46 / 0. This is the corrected RC0 — the submitted-PDF "T.serial stalls on barrier
sync" claim is wrong on sm_80 as well, exactly as on the Ada (sm_89) and GH200 (sm_90) sweeps.
(`max_reduction`, the recovered kernel, is the healthy contrast: 36 regs, 0 spill, **94.3% DRAM**,
73.3% occ — a properly memory-bandwidth-bound kernel.)

**RC3 (corrected: spill is TileLang-LayerNorm-specific, not a conv problem) reproduces.** conv2d
Triton spills **0** bytes (214 regs fit) — confirming the conv gap is occupancy/coalescing-bound, not
spill. The register spill is a *TileLang authoring* effect localized to the wide-fragment LayerNorm:
layer_norm TL spills **51.5 GB ld / 34.4 GB st at 254 regs**, crushing occupancy to 12.4%. matmul
(cuBLAS and Triton) spill 0. This matches the Ada (51.5 GB) and GH200 (45.1 GB) LayerNorm spill — the
RC3 instance is arch-independent for the LayerNorm idiom.

**The logsumexp finding (this run RESOLVES the open cross-architecture question).** On the GH200 (sm_90)
the optimized `logsumexp_opt` register-spills catastrophically (255 regs, ~140 GB ld + ~140 GB st local
traffic, 12.4% occ, 1.0% E_lib) and does **not** transfer to Hopper. **On the A100 (sm_80) it does NOT
spill: 0 bytes local load, 0 bytes local store, 93 regs/thread, 30.5% occupancy, 100% global-load
efficiency, 93.3% DRAM throughput — a cleanly memory-bandwidth-bound kernel running at 0.37 ms.** Its
re-timed efficiency (`mitigation_retime.csv`) is **582.5% of PyTorch**, confirmed genuine by the
zero-spill counters. The catastrophic spill is therefore **sm_90-specific** (the wider register file
pressure / scheduling on Hopper), not a property of the kernel: on sm_80 and sm_89 the same source is a
true authoring-artifact recovery; on sm_90 it is an RC3-class residual. This settles the open item left
by the GH200 sweep — the A100 `tab:mitigation` LogSumExp number is real, not a spill artifact.

**RC1 (vectorization).** layer_norm TL loads at 50% efficiency (1 sector/req, scalar-ish); argmax at
12.5%; conv2d Triton at 26.3% (the coalescing side of the conv residual). `max_reduction` and
`logsumexp_opt` load at **100%** — the recovered kernels vectorize cleanly.

## How this lands in the paper

- Feeds the **A100 column / caption** of `tab:rootcauses` (`tex/analysis.tex`): corrected RC0 (barrier=0,
  long_scoreboard dominates) and RC3 (TileLang-LN 51.5/34.4 GB spill, conv2d 0 spill) confirmed on the
  primary architecture, consistent with Ada and GH200.
- **Resolves the `% TODO(verify)` on the A100 `tab:mitigation` LogSumExp 581.7%/582.5%** (`tex/mitigation.tex`): sm_80 confirmed **not** to spill (0 local traffic, 93 regs,
  30.5% occ — `ncu_summary.csv`), so the number is genuine. The spill is sm_90-only — keep the dual-arch
  caveat ("every family kernel but `logsumexp` on the GH200"), now grounded by both sweeps.
- `logsumexp:tilelang_opt:large` is a permanent ncu target on every arch; the A100 vs GH200 contrast
  (0 GB vs ~280 GB local spill for the same source) is the clean architecture-specific non-transfer.
