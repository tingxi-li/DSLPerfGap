#!/usr/bin/env python3
"""
Evaluation harness for all KernelBench_dedup kernels.

Auto-discovers kernels across level1, level2, level3, and tritonbench,
imports each kernel's unified interface (run / get_test_inputs), and
verifies they execute without error on CUDA.

Usage:
    python eval_all.py                              # run everything
    python eval_all.py --level level1               # single level
    python eval_all.py --kernel 19_relu             # single kernel (substring match)
    python eval_all.py --output results.json        # custom output path
    python eval_all.py --timeout 120                # per-kernel timeout (seconds)
    python eval_all.py --list                       # just list discovered kernels
"""

import argparse
import gc
import importlib.util
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
LEVELS = ["level1", "level2", "level3", "tritonbench"]


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_kernels(levels=None, kernel_filter=None):
    """Scan directories and return list of (level, name, path) tuples."""
    levels = levels or LEVELS
    kernels = []
    for level in levels:
        level_dir = BASE_DIR / level
        if not level_dir.is_dir():
            continue
        for kdir in sorted(level_dir.iterdir()):
            if not kdir.is_dir():
                continue
            impl = kdir / "pytorch_impl.py"
            if not impl.exists():
                continue
            name = kdir.name
            if kernel_filter and kernel_filter not in name:
                continue
            kernels.append((level, name, kdir))
    return kernels


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic import
# ═══════════════════════════════════════════════════════════════════════════════

def load_module(kdir):
    """Import pytorch_impl.py from a kernel directory."""
    fpath = kdir / "pytorch_impl.py"
    # Add kernel dir to sys.path for local imports
    kdir_str = str(kdir)
    if kdir_str not in sys.path:
        sys.path.insert(0, kdir_str)
    spec = importlib.util.spec_from_file_location(
        f"pytorch_impl_{kdir.name}", str(fpath)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════════════
# Output validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_output(output):
    """Check that output tensors contain no NaN/Inf."""
    if output is None:
        return True, "output is None (no-op kernel)"
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            return False, "output contains NaN"
        if torch.isinf(output).any():
            return False, "output contains Inf"
        return True, f"shape={list(output.shape)}, dtype={output.dtype}"
    if isinstance(output, (tuple, list)):
        for i, t in enumerate(output):
            if isinstance(t, torch.Tensor):
                if torch.isnan(t).any():
                    return False, f"output[{i}] contains NaN"
                if torch.isinf(t).any():
                    return False, f"output[{i}] contains Inf"
        return True, f"{len(output)} outputs"
    return True, f"type={type(output).__name__}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════════════

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_ERROR = "ERROR"
STATUS_OOM = "OOM"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_IMPORT_ERROR = "IMPORT_ERROR"


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Kernel execution timed out")


def eval_kernel(level, name, kdir, timeout=120):
    """Evaluate a single kernel. Returns a result dict."""
    record = {
        "level": level,
        "kernel": name,
        "status": STATUS_ERROR,
        "time_ms": None,
        "output_info": None,
        "error": None,
    }

    # Import
    try:
        mod = load_module(kdir)
    except Exception as e:
        record["status"] = STATUS_IMPORT_ERROR
        record["error"] = f"{type(e).__name__}: {e}"
        return record

    # Check interface
    if not hasattr(mod, "run"):
        record["status"] = STATUS_IMPORT_ERROR
        record["error"] = "Missing run() function"
        return record

    # Execute with timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        t0 = time.perf_counter()
        output = mod.run()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        signal.alarm(0)  # cancel alarm

        valid, info = validate_output(output)
        record["time_ms"] = round(elapsed_ms, 2)
        record["output_info"] = info
        record["status"] = STATUS_PASS if valid else STATUS_FAIL
        if not valid:
            record["error"] = info

    except TimeoutError:
        record["status"] = STATUS_TIMEOUT
        record["error"] = f"Exceeded {timeout}s timeout"
    except torch.cuda.OutOfMemoryError:
        record["status"] = STATUS_OOM
        record["error"] = "CUDA out of memory"
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        err_str = str(e).lower()
        if "out of memory" in err_str:
            record["status"] = STATUS_OOM
            record["error"] = f"{type(e).__name__}: {e}"
            torch.cuda.empty_cache()
            gc.collect()
        else:
            record["status"] = STATUS_ERROR
            record["error"] = f"{type(e).__name__}: {e}"
            record["traceback"] = traceback.format_exc()
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

    return record


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

STATUS_SYMBOLS = {
    STATUS_PASS: "\033[92m✓\033[0m",
    STATUS_FAIL: "\033[91m✗\033[0m",
    STATUS_ERROR: "\033[91mE\033[0m",
    STATUS_OOM: "\033[93mM\033[0m",
    STATUS_TIMEOUT: "\033[93mT\033[0m",
    STATUS_IMPORT_ERROR: "\033[91mI\033[0m",
}


def print_result(record):
    sym = STATUS_SYMBOLS.get(record["status"], "?")
    time_str = f"{record['time_ms']:.1f}ms" if record["time_ms"] else "—"
    info = record.get("output_info", "") or ""
    err = ""
    if record["status"] not in (STATUS_PASS,):
        err = f" | {record.get('error', '')}"
    print(f"  {sym} {record['level']:12s} {record['kernel']:50s} {time_str:>10s} {info}{err}")


def print_summary(results):
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    total = len(results)
    print(f"\n{'='*80}")
    print(f"Summary: {total} kernels")
    for status in [STATUS_PASS, STATUS_FAIL, STATUS_ERROR, STATUS_OOM, STATUS_TIMEOUT, STATUS_IMPORT_ERROR]:
        c = counts.get(status, 0)
        if c > 0:
            print(f"  {status:15s}: {c:4d}  ({100*c/total:.1f}%)")
    print(f"{'='*80}")

    # Per-level breakdown
    level_counts = {}
    for r in results:
        lv = r["level"]
        if lv not in level_counts:
            level_counts[lv] = {"total": 0, "pass": 0}
        level_counts[lv]["total"] += 1
        if r["status"] == STATUS_PASS:
            level_counts[lv]["pass"] += 1
    print("\nPer-level:")
    for lv in LEVELS:
        if lv in level_counts:
            lc = level_counts[lv]
            print(f"  {lv:12s}: {lc['pass']}/{lc['total']} pass")


def save_results(results, output_path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(results),
            "pass": sum(1 for r in results if r["status"] == STATUS_PASS),
            "fail": sum(1 for r in results if r["status"] == STATUS_FAIL),
            "error": sum(1 for r in results if r["status"] == STATUS_ERROR),
            "oom": sum(1 for r in results if r["status"] == STATUS_OOM),
            "timeout": sum(1 for r in results if r["status"] == STATUS_TIMEOUT),
            "import_error": sum(1 for r in results if r["status"] == STATUS_IMPORT_ERROR),
        },
        "results": results,
    }
    # Remove traceback from JSON (too verbose)
    for r in data["results"]:
        r.pop("traceback", None)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate KernelBench_dedup kernels")
    parser.add_argument("--level", choices=LEVELS, help="Run only this level")
    parser.add_argument("--kernel", type=str, help="Filter kernels by substring match")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "eval_all.json"),
                        help="Output JSON path")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-kernel timeout in seconds (default: 120)")
    parser.add_argument("--list", action="store_true", help="Just list discovered kernels")
    args = parser.parse_args()

    levels = [args.level] if args.level else None
    kernels = discover_kernels(levels=levels, kernel_filter=args.kernel)

    if args.list:
        for level, name, kdir in kernels:
            print(f"  {level:12s}  {name}")
        print(f"\nTotal: {len(kernels)} kernels")
        return

    print(f"Evaluating {len(kernels)} kernels (timeout={args.timeout}s)")
    print(f"{'='*80}")

    results = []
    for i, (level, name, kdir) in enumerate(kernels, 1):
        print(f"[{i}/{len(kernels)}]", end="")
        record = eval_kernel(level, name, kdir, timeout=args.timeout)
        results.append(record)
        print_result(record)

        # Clean up GPU memory between kernels
        torch.cuda.empty_cache()
        gc.collect()

    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
