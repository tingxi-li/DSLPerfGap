# Tuning-sweep methodology fix + corrected H100 results

**GPU:** NVIDIA H100 80GB HBM3 (sm_90) · **Date:** 2026-06-24 · stack `torch==2.8.0+cu126 / triton==3.4.0 / tilelang==0.1.6.post1`

This documents (1) a methodological bug in `ViperBench/tuning/sweep.py`, (2) the fix,
and (3) the corrected "does heuristic tuning help?" result on H100 — which **contradicts
the §5 claim that tuning yields Δ≈0pp across the 22 kernels**.

## 1. The bug (now fixed)

`sweep.py::sweep_kernel()` applied each candidate config with `mod._TUNED = config`
*after* `load_module()`. But the ViperBench impls read `_TUNED` at **import time** to
build the kernel (TileLang `@tilelang.jit` kernels are constructed at import). So
reassigning `mod._TUNED` afterward is a **no-op** — every candidate timed the *same*
import-default kernel, and the "winner" was pure measurement noise (compounded by
`WARMUP=3, TRIALS=10` wall-clock timing). The noise-selected config was then written to
`tuning_cache.json`, and `benchmark_tuned.py` *did* apply it (fresh import) — producing a
"tuned" profile full of spurious deltas. The sweep also never considered the impl's
hardcoded **default** as a candidate, so the selected config could be worse than default.

## 2. The fix

Three changes to `sweep.py` (this branch):

1. **Config applied at build time** — each candidate is installed by patching
   `tuning.cache.get_best_config` *before* a fresh import, so the kernel is actually
   (re)built with that config. (The post-import `_TUNED` assignment is gone.)
2. **CUDA-event timing, `WARMUP=15 / TRIALS=50`** (was wall-clock 3/10).
3. **The hardcoded default is a candidate** (`config=None`); the cache is overridden
   **only** when a grid config genuinely beats the default at the tuning shape — so a
   sweep can never select something worse than default. When the default wins, no cache
   entry is written (the impl falls back to default → honest Δ=0 for that kernel).

**Validation** (`matmul`, tuning shape 4096²): configs now genuinely vary
(0.40→2.39 ms across the grid). `matmul/tilelang` correctly **keeps the default** (no
override) — the broken sweep had written the *worst* config here (the +462% regression).
`matmul/triton` picks a genuinely faster config (−13%).

Result on H100: **32 of 44** kernel/impl pairs got a real override; **12 kept the
default** (incl. `matmul/tilelang`).

## 3. Corrected result — tuning is NOT Δ≈0; it is a wide, shape-dependent spread

Comparison: `benchmark_tuned.py` (tuned rows, measured now, cache-on) vs the committed
untuned base `ViperBench/results/profile.H100-80GB-HBM3.csv` (genuine defaults, prior
session). Full 5-impl profile: `ViperBench/results/profile.H100-80GB-HBM3.tuned.csv`.

### Cross-session noise floor (non-override rows: default-now vs default-prior-session)

- n=24 comparisons; median |Δ| = **1.5%**, p90 = **5.3%**, max = **56.7%**
- single outlier: `softmax/triton/small` at −56.7% (sub-0.06 ms kernel, launch-overhead variance)
- **signal threshold used: |Δ| > 15%** (well above p90); headline effects below exceed even the 56.7% outlier.

### Genuine tuning effect (32 overrides, 62 comparisons)

- **Geomean tuned/untuned = 0.822 (−17.8% net)** — tuning helps *on average*…
- …but **bidirectional**: **20 improve >15%, 9 regress >15%, 33 neutral**.

**Large-shape genuine improvements**

| Δ% | kernel | impl | untuned ms | tuned ms |
|---:|---|---|---:|---:|
| −90.1% | argmax | triton | 12.2624 | 1.2193 |
| −78.3% | linear_activation | triton | 37.1426 | 8.0752 |
| −74.2% | layer_norm | tilelang | 193.6155 | 49.9810 |
| −73.9% | rms_norm | tilelang | 159.3898 | 41.6175 |
| −57.5% | logsumexp | tilelang | 0.6217 | 0.2642 |
| −30.0% | leaky_relu | triton | 12.6735 | 8.8694 |
| −23.5% | matrix_transpose | triton | 0.5253 | 0.4020 |
| −22.6% | matmul | triton | 73.1432 | 56.6351 |

**Large-shape genuine regressions**

| Δ% | kernel | impl | untuned ms | tuned ms |
|---:|---|---|---:|---:|
| +21.3% | softmax | tilelang | 1.2742 | 1.5462 |
| +32.3% | swiglu | triton | 0.1713 | 0.2267 |
| +35.7% | matmul | tilelang | 28.9469 | 39.2686 |
| +94.7% | conv2d | tilelang | 18.5937 | 36.1943 |
| +135.1% | log_softmax | tilelang | 1.2647 | 2.9728 |
| +188.4% | swiglu | tilelang | 0.5435 | 1.5676 |

### Why the regressions — shape transfer, not noise

`sweep.py` tunes on **one shape per kernel** (small/medium, e.g. matmul 4096², norms
512×1024 — see `_make_inputs`), but the single cached config is applied to **all** shapes,
including large. Configs best at the tuning shape do not transfer to large: the worst
cases (`swiglu`/`log_softmax`/`conv2d` TileLang at large) are genuine — the swept config
is real, just wrong for the deployment shape. This is the §7.3 / W7 shape-sensitivity
effect, made visible only after the sweep was fixed.

The large TileLang norm/reduction *wins* (`layer_norm` −74%, `rms_norm` −74%) mirror the
AKO4ALL mitigation campaigns (`tab:mitigation`): the same kernels where better block/stage
choices pay off massively.

## 4. Paper implication (author decision)

- **§5 "heuristic tuning → Δ≈0pp across 22 kernels"** (`tex/evaluation.tex`) rests on this
  broken sweep. The Δ=0 it reported is the *signature of the bug* (every candidate timed
  the default → cache often left at default → tuned==untuned). A **correct** sweep yields a
  wide bidirectional, shape-dependent spread (−90% … +188% at large; −17.8% geomean), **not** Δ=0.
- The §5 measurement was on **Ada**; this corrected run is on **H100**. To close the loop,
  re-run the *fixed* sweep on Ada (RTX 4000 Ada) and re-derive the §5 number. The fix is
  arch-general (`sweep.py`), so the Ada cache will need regenerating too. Until then, treat
  the §5 Δ=0 claim as **unsupported**.
- Relevant to **W7** (autotune reconciliation, §5 vs §7.3) — this *is* the reconciliation:
  single-shape tuning helps at the tuned shape and can hurt at others.

## 5. Replay

```bash
cd ViperBench && python -m tuning.sweep --all          # fixed sweep -> tuning_cache.json (H100 keys)
# stage untuned base, then measure tuned rows:
cp results/profile.H100-80GB-HBM3.csv results/profile.csv
python benchmark_tuned.py                               # adds *_tuned rows
cp results/profile.csv results/profile.H100-80GB-HBM3.tuned.csv
# restore canonical profile.csv afterward (it is the Ada baseline)
```

**Caveat (this box):** unprivileged Vast container → no clock lock (`nvidia-smi -lgc`
denied), so timings are unlocked-clock; cross-session base/tuned comparison carries the
~1.5% median (5.3% p90) noise quantified above. Headline effects (>20%) are far above it.
