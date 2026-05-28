"""
Portable experiment harness for ASE-2026 #4134 rebuttal experiments.

Design goals:
  * IDENTICAL code runs on RTX 4000 Ada (now) and A100/H100 (later).
    Nothing is hardcoded to sm_89 — device properties are queried at runtime.
  * Every result is tagged by the detected GPU so Ada / A100 / H100 outputs
    never collide:  experiments/results/<gpu_slug>/<experiment>.csv
  * Timing reports median AND mean+/-std over repeated reps (gives the
    confidence intervals reviewers asked about for "is 94.6% vs 97.8%
    meaningful?", W8), measured on an idle GPU with CUDA events.

Usage in an experiment script:
    from _harness import device_info, time_kernel, load_kernel, write_csv, banner
"""
import csv
import importlib.util
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
VIPER_DIR = REPO_ROOT / "ViperBench"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# Make ViperBench importable so kernel impls can do `from tuning.cache import ...`
if str(VIPER_DIR) not in sys.path:
    sys.path.insert(0, str(VIPER_DIR))


# ---------------------------------------------------------------------------
# Device identity / properties  (queried, never hardcoded -> portable)
# ---------------------------------------------------------------------------
def device_slug(idx: int = 0) -> str:
    """Filesystem-safe tag for the current GPU, e.g. 'NVIDIA_RTX_4000_Ada' or
    'NVIDIA_A100-SXM4-80GB'. Used to namespace result directories per arch."""
    name = torch.cuda.get_device_name(idx)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def device_info(idx: int = 0) -> dict:
    """All hardware facts the experiments need, queried at runtime.
    Returns L2 size, memory, SM count, compute capability, sw versions."""
    p = torch.cuda.get_device_properties(idx)
    cc = f"{p.major}.{p.minor}"
    return {
        "gpu_name": torch.cuda.get_device_name(idx),
        "gpu_slug": device_slug(idx),
        "sm": f"sm_{p.major}{p.minor}",
        "compute_capability": cc,
        "sm_count": p.multi_processor_count,
        "total_mem_GB": round(p.total_memory / 1024**3, 2),
        # L2_cache_size exists on modern torch; guard for portability.
        "l2_cache_MB": round(getattr(p, "L2_cache_size", 0) / 1024**2, 2),
        "torch": torch.version.__version__,
        "cuda": torch.version.cuda,
        "driver": _driver_version(),
        "host": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _driver_version() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def banner(experiment: str, idx: int = 0):
    info = device_info(idx)
    print("=" * 72)
    print(f"  Experiment: {experiment}")
    print(f"  GPU: {info['gpu_name']}  ({info['sm']}, {info['sm_count']} SMs, "
          f"{info['total_mem_GB']} GB, L2={info['l2_cache_MB']} MB)")
    print(f"  torch {info['torch']} / cuda {info['cuda']} / driver {info['driver']}")
    print("=" * 72)
    return info


# ---------------------------------------------------------------------------
# Timing  (CUDA-event based; median + mean/std for confidence intervals)
# ---------------------------------------------------------------------------
def time_kernel(fn, *args, warmup: int = 15, reps: int = 50, **kwargs) -> dict:
    """Time fn(*args, **kwargs) on the current GPU.
    Returns median_ms, mean_ms, std_ms, p95_ms, min_ms, reps.
    Run only on an idle GPU; serialize timing experiments."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        starter.record()
        fn(*args, **kwargs)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms

    times.sort()
    n = len(times)
    mean = sum(times) / n
    std = (sum((t - mean) ** 2 for t in times) / n) ** 0.5
    return {
        "median_ms": round(times[n // 2], 5),
        "mean_ms": round(mean, 5),
        "std_ms": round(std, 5),
        "p95_ms": round(times[min(n - 1, int(0.95 * n))], 5),
        "min_ms": round(times[0], 5),
        "reps": reps,
    }


def library_efficiency(t_lib_ms: float, t_dsl_ms: float) -> float:
    """E_lib = t_library / t_DSL * 100 (paper's primary metric)."""
    if t_dsl_ms <= 0:
        return float("nan")
    return round(100.0 * t_lib_ms / t_dsl_ms, 2)


# ---------------------------------------------------------------------------
# Kernel loading  (reuse ViperBench's per-kernel impl files unchanged)
# ---------------------------------------------------------------------------
def load_impl(kernel: str, impl: str):
    """Import ViperBench/<kernel>/<impl>_impl.py and return the module.
    impl in {'pytorch','triton','tilelang'}. The unified-API function is
    getattr(mod, <fn>) where <fn> is usually the kernel name."""
    path = VIPER_DIR / kernel / f"{impl}_impl.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"{kernel}_{impl}_impl", str(path))
    mod = importlib.util.module_from_spec(spec)
    # impls expect their own dir + ViperBench on path
    kdir = str(VIPER_DIR / kernel)
    for p in (kdir, str(VIPER_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec.loader.exec_module(mod)
    return mod


def load_optimized(name: str):
    """Import an AKO4ALL mitigation kernel, e.g. 'layer_norm_tilelang'."""
    path = REPO_ROOT / "AKO4ALL" / "results" / "optimized" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Result output  (namespaced per GPU arch)
# ---------------------------------------------------------------------------
def results_dir(idx: int = 0) -> Path:
    d = RESULTS_ROOT / device_slug(idx)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_csv(experiment: str, rows: list, fieldnames: list, idx: int = 0) -> Path:
    """Write rows to experiments/results/<gpu_slug>/<experiment>.csv,
    prepending GPU identity columns so every row is self-describing."""
    info = device_info(idx)
    meta_cols = ["gpu_name", "sm", "timestamp_utc"]
    full_fields = meta_cols + fieldnames
    out = results_dir(idx) / f"{experiment}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=full_fields)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r.update(gpu_name=info["gpu_name"], sm=info["sm"],
                     timestamp_utc=info["timestamp_utc"])
            w.writerow(r)
    print(f"  -> wrote {len(rows)} rows to {out}")
    return out


if __name__ == "__main__":
    # Self-test: print device info + time a trivial op (proves portability layer).
    banner("_harness self-test")
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    print("  matmul timing:", time_kernel(lambda: x @ x, warmup=5, reps=10))
    import json
    print(json.dumps(device_info(), indent=2))
