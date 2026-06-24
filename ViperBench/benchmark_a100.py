#!/usr/bin/env python3
"""
Generate a GPU-consistent profile for the CURRENT device, written to
results/profile.<device_arch>.tuned.csv (matching the existing A100/H100
cross-arch naming) — WITHOUT touching the RTX 4000 Ada baseline in
results/profile.csv (the paper's primary RQ1 data; see paper §3 methodology).

For each (kernel, size) it emits all five impl rows used by the eval tables:
  pytorch | triton | triton_tuned | tilelang | tilelang_tuned

"Untuned" triton/tilelang are profiled against an EMPTY cache, so the impls fall
back to their hardcoded defaults. The "_tuned" rows use the real on-disk tuning
cache (the swept config for this device's arch), exactly as benchmark_tuned.py
deploys it. The cache redirection reuses the same import-time mechanism the fixed
tuning sweep relies on (set tuning.cache.CACHE_PATH, then import the impl fresh).

Test cases, timing, and peak-memory logic are reused verbatim from
benchmark_tuned.py so the numbers are directly comparable to that script.

Usage (run from inside ViperBench/):
    python benchmark_a100.py
"""
import csv
import sys
import tempfile
from pathlib import Path

import torch

BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR))

import benchmark_tuned as bt
from tuning import cache as cache_mod
from tuning.cache import get_gpu_arch

# Real cache (swept configs) vs an empty cache (forces hardcoded defaults).
REAL_CACHE = cache_mod.CACHE_PATH
_EMPTY_DIR = Path(tempfile.mkdtemp(prefix="empty_cache_"))
EMPTY_CACHE = _EMPTY_DIR / "empty.json"
EMPTY_CACHE.write_text("{}")

# (impl module, output label, cache to read at import)
PLAN = [
    ("pytorch_impl",  "pytorch",        EMPTY_CACHE),   # reference; ignores the cache anyway
    ("triton_impl",   "triton",         EMPTY_CACHE),   # untuned = defaults
    ("triton_impl",   "triton_tuned",   REAL_CACHE),    # tuned = swept config
    ("tilelang_impl", "tilelang",       EMPTY_CACHE),
    ("tilelang_impl", "tilelang_tuned", REAL_CACHE),
]


def _profile(kdir, module_name, fn_name, args, kwargs, warmup, iters, cache_path):
    cache_mod.CACHE_PATH = cache_path          # impls read this AT IMPORT
    mod = bt.load_module(kdir, module_name)
    fn = getattr(mod, fn_name)
    torch.cuda.empty_cache()
    return bt.profile_fn(fn, args, kwargs, warmup=warmup, iters=iters)


def main():
    arch = get_gpu_arch()
    out_csv = BENCH_DIR / "results" / f"profile.{arch}.tuned.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cases = bt.get_test_cases()
    kernel_dirs = sorted(
        d for d in BENCH_DIR.iterdir()
        if d.is_dir() and (d / "triton_impl.py").exists()
    )

    fieldnames = ["kernel", "size", "impl", "input_desc", "latency_ms", "peak_memory_mb"]
    rows = []
    print(f"Device: {torch.cuda.get_device_name(0)}  (arch key: {arch})")
    print(f"Output: {out_csv}\n")
    print(f"{'Kernel':<20}{'Size':<6}{'Impl':<16}{'Lat(ms)':>11}{'Mem(MB)':>11}  Input")
    print("-" * 100)

    for kdir in kernel_dirs:
        name = kdir.name
        if name not in cases:
            continue
        for (size_label, fn_name, args, kwargs, desc, is_slow) in cases[name]:
            kwargs = kwargs or {}
            iters = bt.MEASURE_ITERS_SLOW if is_slow else bt.MEASURE_ITERS
            warmup = 3 if is_slow else bt.WARMUP_ITERS
            for module_name, impl_label, cache_path in PLAN:
                if not (kdir / f"{module_name}.py").exists():
                    continue
                try:
                    lat, mem = _profile(kdir, module_name, fn_name, args, kwargs,
                                        warmup, iters, cache_path)
                    rows.append({"kernel": name, "size": size_label, "impl": impl_label,
                                 "input_desc": desc, "latency_ms": round(lat, 4),
                                 "peak_memory_mb": round(mem, 2)})
                    print(f"{name:<20}{size_label:<6}{impl_label:<16}{lat:>11.4f}{mem:>11.2f}  {desc}")
                except Exception as e:
                    rows.append({"kernel": name, "size": size_label, "impl": impl_label,
                                 "input_desc": desc, "latency_ms": "ERROR", "peak_memory_mb": "ERROR"})
                    print(f"{name:<20}{size_label:<6}{impl_label:<16}{'ERROR':>11}  {str(e)[:60]}")
        print()

    cache_mod.CACHE_PATH = REAL_CACHE
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    n_err = sum(1 for r in rows if r["latency_ms"] == "ERROR")
    print(f"\nWrote {len(rows)} rows ({n_err} ERROR) to {out_csv}")


if __name__ == "__main__":
    main()
