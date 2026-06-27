#!/usr/bin/env python3
"""Experiment - Tuning-shape pitfall: small-shape vs large-shape autotuning.

ViperBench's autotuning sweep (tuning/sweep.py) selects a config by timing the
candidate grid at a per-kernel TUNING shape (`_make_inputs`). For several kernels
that shape is far smaller than the benchmark "large" shape the kernel actually
runs at (e.g. batched_matmul tunes at A:64x128 but runs at A:128x2048;
attention tunes at 1x2x64x32 but runs at 8x32x2048x128). At the small shape the
candidate configs are within measurement noise of each other, so the "winner" is
noise-selected and can be *worse than the hardcoded default* at the real shape.

This experiment quantifies the tradeoff: for each affected kernel we tune the
SAME grid at (a) the small tuning shape and (b) the large benchmark shape, then
evaluate the selected config -- and the default -- at the large shape. We report
the config quality (latency at the shape that runs) against the tuning OVERHEAD
(wall-clock to sweep the grid), since large-shape tuning is more expensive per
candidate.

Reuses tuning/sweep.py's exact selection machinery (per-config get_best_config
patch + module reload + CUDA-event time_fn) so the comparison is apples-to-apples
with the production sweep.

Usage:
    python exp_tuning_shape_tradeoff.py            # full (locked clocks recommended)
    python exp_tuning_shape_tradeoff.py --smoke    # 1 kernel, fewer reps
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ViperBench"))
import _harness as H  # noqa: E402
import tuning.sweep as S  # noqa: E402
import tuning.cache as _cachemod  # noqa: E402
from tuning.configs import TRITON_CONFIGS, TILELANG_CONFIGS  # noqa: E402

# The degradation cases (small-shape sweep picked a config that loses at large).
# large_inputs mirror benchmark.py:get_test_cases() "large"; small_inputs mirror
# tuning/sweep.py:_make_inputs (the production tuning shape).
def _f16(*s):
    return torch.randn(*s, device="cuda", dtype=torch.float16)
def _f32(*s):
    return torch.randn(*s, device="cuda", dtype=torch.float32)

CASES = [
    dict(kernel="batched_matmul", impl="triton",
         small=lambda: (_f16(64, 128), _f16(64, 128, 128)),
         large=lambda: (_f16(128, 2048), _f16(128, 2048, 2048)),
         small_desc="A:64x128", large_desc="A:128x2048"),
    dict(kernel="mean_reduction", impl="triton",
         small=lambda: (_f32(1024, 1024), 1),
         large=lambda: (_f32(8192, 32768), 1),
         small_desc="1024x1024", large_desc="8192x32768"),
    dict(kernel="attention", impl="triton",
         small=lambda: (_f32(1, 2, 64, 32), _f32(1, 2, 64, 32), _f32(1, 2, 64, 32)),
         large=lambda: (_f32(8, 32, 2048, 128), _f32(8, 32, 2048, 128), _f32(8, 32, 2048, 128)),
         small_desc="1x2x64x32", large_desc="8x32x2048x128"),
]


def _fn_name(kernel):
    return S.KERNEL_FN_NAMES.get(kernel) or (kernel + "_fwd" if kernel == "attention" else kernel)


def tune_grid(kernel, impl, configs, inputs, warmup, trials):
    """Replicate sweep_kernel: time [default]+grid at `inputs`, return (best_cfg, best_lat, wall_s, n)."""
    kdir = S.BENCH_DIR / kernel
    fn_name = _fn_name(kernel)
    candidates = [None] + list(configs)
    best_cfg, best_lat = None, float("inf")
    orig = _cachemod.get_best_config
    t0 = time.time()
    for cfg in candidates:
        try:
            _cachemod.get_best_config = (lambda *a, _c=cfg, **k: _c)
            mod = S.load_module(kdir, f"{impl}_impl")
            fn = getattr(mod, fn_name)
            torch.cuda.empty_cache()
            lat = S.time_fn(fn, inputs, warmup=warmup, trials=trials)
            if lat < best_lat:
                best_lat, best_cfg = lat, cfg
        except Exception as e:
            print(f"    cfg {cfg} ERROR: {str(e)[:70]}")
        finally:
            _cachemod.get_best_config = orig
    return best_cfg, best_lat, time.time() - t0, len(candidates)


def eval_at(kernel, impl, config, inputs, warmup, trials):
    """Time one specific config at `inputs` (large), median ms."""
    kdir = S.BENCH_DIR / kernel
    fn_name = _fn_name(kernel)
    orig = _cachemod.get_best_config
    try:
        _cachemod.get_best_config = (lambda *a, _c=config, **k: _c)
        mod = S.load_module(kdir, f"{impl}_impl")
        fn = getattr(mod, fn_name)
        torch.cuda.empty_cache()
        return S.time_fn(fn, inputs, warmup=warmup, trials=trials)
    finally:
        _cachemod.get_best_config = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cases = CASES[:1] if args.smoke else CASES
    twarm, ttrial = (5, 10) if args.smoke else (15, 50)   # tuning reps (match sweep: 15/50)
    ewarm, etrial = (5, 20) if args.smoke else (15, 100)  # eval reps (more, for the reported latency)
    arch = _cachemod.get_gpu_arch()
    cache = _cachemod.load_cache()

    print(f"== tuning-shape tradeoff == {H.device_info()['gpu_name']}  (arch={arch})")
    rows = []
    for c in cases:
        k, impl = c["kernel"], c["impl"]
        configs = (TRITON_CONFIGS if impl == "triton" else TILELANG_CONFIGS).get(k, [])
        cached_cfg = cache.get(f"{k}/{impl}/{arch}")  # what the production (small-shape) sweep actually cached
        print(f"\n### {k} ({impl}) — {len(configs)} grid configs + default; production-cached cfg={cached_cfg}")
        small_in = c["small"](); large_in = c["large"]()

        # Tuning COST: wall-clock to sweep the grid at each shape (the overhead).
        s_pick, _, s_wall, n = tune_grid(k, impl, configs, small_in, twarm, ttrial)
        l_cfg, _, l_wall, _ = tune_grid(k, impl, configs, large_in, twarm, ttrial)
        del small_in

        # Config QUALITY: evaluate at the LARGE shape that actually runs.
        def_lat = eval_at(k, impl, None, large_in, ewarm, etrial)        # no tuning
        prod_lat = eval_at(k, impl, cached_cfg, large_in, ewarm, etrial)  # production small-shape pick (the artifact)
        l_lat = eval_at(k, impl, l_cfg, large_in, ewarm, etrial)         # large-shape re-tune
        del large_in

        row = dict(
            kernel=k, impl=impl, n_candidates=n,
            tune_small_shape=c["small_desc"], tune_large_shape=c["large_desc"],
            default_large_ms=round(def_lat, 4),
            cached_cfg=str(cached_cfg), cached_large_ms=round(prod_lat, 4),
            cached_vs_default_pct=round(100 * (prod_lat - def_lat) / def_lat, 1),
            large_cfg=str(l_cfg), large_tuned_large_ms=round(l_lat, 4),
            large_vs_default_pct=round(100 * (l_lat - def_lat) / def_lat, 1),
            retune_gain_x=round(prod_lat / l_lat, 2),  # speedup from re-tuning at large shape
            small_tune_s=round(s_wall, 1), large_tune_s=round(l_wall, 1),
            tune_overhead_ratio=round(l_wall / s_wall, 1) if s_wall > 0 else None,
            small_shape_rerun_pick=str(s_pick),  # illustrates small-shape pick is noisy/non-reproducible
        )
        rows.append(row)
        print(f"  default      @large: {def_lat:.4f} ms")
        print(f"  CACHED (small-tune): {cached_cfg} -> @large {prod_lat:.4f} ms ({row['cached_vs_default_pct']:+}% vs default)")
        print(f"  RE-TUNED (large):    {l_cfg} -> @large {l_lat:.4f} ms ({row['large_vs_default_pct']:+}% vs default)")
        print(f"  => re-tuning at large recovers {row['retune_gain_x']}x;  tuning cost {s_wall:.0f}s(small) -> {l_wall:.0f}s(large) = {row['tune_overhead_ratio']}x overhead")
        print(f"     (note: a fresh small-shape re-tune this run picked {s_pick} -- small-shape pick is noise-driven, not reproducible)")

    out = H.write_csv("tuning_shape_tradeoff", rows, list(rows[0].keys()))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
