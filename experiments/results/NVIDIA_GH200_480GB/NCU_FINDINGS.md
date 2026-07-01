# Hardware-counter findings — GH200 480GB (sm_90, Hopper)

Canonical RC interpretation of the GH200 Nsight Compute sweep (`ncu_counters.sh` → `ncu/` →
`consolidate_ncu.py` → `ncu_summary.csv`). Counters are the same arch-portable sets used on the
RTX 4000 Ada (sm_89) and A100 runs, so the taxonomy is directly comparable across architectures.
Each kernel is profiled at the paper "large" shape, exactly one launch inside an NVTX `TARGET` range.

## The table

| metric (RC) | matmul cuBLAS | matmul Triton | conv2d Triton | layer_norm TL | argmax TL | max_red TL | logsumexp TL_opt |
|---|--:|--:|--:|--:|--:|--:|--:|
| kernel | `nvjet_hsh…coopB_NNN` | `matmul_kernel` | `conv2d_forward_kernel` | `func_kernel` | `func_kernel` | `func_kernel` | `func_kernel` |
| **RC3** regs/thread | 168 | 128 | 226 | **254** | 32 | 52 | **255** |
| **RC3** achieved occ % | 14.8 | 18.7 | 12.5 | **12.1** | 24.1 | 48.8 | **12.4** |
| **RC3** spill load B | 0 | 0 | 0 | **45.1 GB** | 0 | 0 | **140.2 GB** |
| **RC3** spill store B | 0 | 0 | 0 | **34.4 GB** | 0 | 0 | **140.1 GB** |
| **RC0** stall long_scoreboard | 3.97 | 4.03 | 2.08 | **49.95** | 45.26 | 14.55 | 26.17 |
| **RC0** stall barrier (sync) | 3.28 | 1.84 | 0.54 | **0** | **0** | 6.85 | **0** |
| **RC1** global-load eff % (bytes/sector) | 0\* | 0\* | 12.1 | 50.0 | 12.5 | **100** | 12.5 |
| **RC2b** L2 hit % | 64.6 | 52.2 | 89.5 | 47.5 | 33.3 | **0.6** | 51.1 |
| **RC2b** DRAM throughput % | 41.9 | 86.8 | 5.3 | 43.0 | 4.9 | **93.5** | 63.5 |
| kernel time (ms) | 11.28 | 20.07 | 11.37 | 47.71 | 5.52 | **0.14** | 107.87 |

\* cuBLAS (`nvjet…`) and the Triton GEMM use TMA / `cp.async` bulk loads on Hopper, so the classic
`global_op_ld` bytes/sector counter reads 0 — not a vectorization failure.

## What the counters say

**RC0 (corrected: memory-latency, not barrier sync) reproduces on Hopper.** The slow TileLang
norm/reduction kernels stall on `long_scoreboard` (memory latency), with `barrier`=**0**: layer_norm
49.95 / 0, argmax 45.26 / 0. This is the corrected RC0 — the submitted-PDF "T.serial stalls on barrier
sync" claim is wrong on sm_90 too. (`max_reduction`, the recovered kernel, is the healthy contrast: 52
regs, 0 spill, **93.5% DRAM**, 48.8% occ — a properly memory-bandwidth-bound kernel.)

**RC3 (corrected: spill is TileLang-specific, not a conv problem) reproduces and now has a second
instance.** conv2d Triton spills **0** bytes (226 regs fit) — confirming the conv gap is occupancy/
coalescing-bound, not spill. The register spill is a *TileLang authoring* effect: layer_norm TL spills
45.1 GB ld / 34.4 GB st at 254 regs, and the **optimized logsumexp** kernel spills **140.2 GB ld /
140.1 GB st at 255 regs (the ceiling)**, crushing occupancy to 12.4%. Its 63.5% "DRAM throughput" is
spill traffic thrashing local memory, not useful work — for a ~0.5 GB input, ~280 GB of spill traffic.

**The logsumexp finding (new; the cross-architecture point).** `logsumexp_opt` is reported as recovered
on the A100 (581.7%) but on the GH200 its A100-tuned wide fp32 fragment register-spills catastrophically
→ 1.0% E_lib (107.9 ms). Even retuned to the GH200-optimal fragment width it caps at 63%. It is the one
norm/reduction-family kernel that does **not** transfer to Hopper — an RC3-class residual, not the RC0
authoring idiom.

**RC1 (vectorization).** layer_norm TL loads at 50% efficiency (1 sector/req, scalar-ish); argmax and
logsumexp at 12.5%; `max_reduction` at 100% (the recovered kernel vectorizes cleanly). conv2d Triton at
12.1% — the coalescing side of the conv residual.

## How this lands in the paper

- Feeds the **GH200 column / caption** of `tab:rootcauses` (`tex/analysis.tex`): corrected RC0 (barrier=0,
  long_scoreboard dominates) and RC3 (TileLang-LN 45/34 GB spill) confirmed on a second architecture.
- Backs the **`tab:mitigation` logsumexp footnote** (`tex/mitigation.tex`): "255 regs, ~280 GB local-memory
  spill traffic, 12% occupancy" cites this sweep (`ncu_summary.csv`).
- **Open item:** the A100 `tab:mitigation` LogSumExp 581.7% uses the same kernel; `logsumexp:tilelang_opt:large`
  is now a permanent ncu target, so the next A100 sweep settles whether sm_80 also spills it.
