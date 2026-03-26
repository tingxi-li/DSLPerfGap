#!/usr/bin/env python3
"""
scripts/run_all.py
------------------
Discovers every kernels/<name>/test_kernel.py, runs it, collects results.
Handles up to MAX_RETRIES compilation/test failures per kernel before moving on.

Usage:
    python scripts/run_all.py [--retries 20] [--kernels comma,list]
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

MAX_RETRIES = 20
RESULTS_DIR = Path("tests/results")
KERNELS_DIR = Path("newBench")


def discover_kernels(only=None):
    kernels = sorted(
        d for d in KERNELS_DIR.iterdir()
        if d.is_dir() and (d / "test_kernel.py").exists()
    )
    if only:
        requested = set(only.split(","))
        kernels = [k for k in kernels if k.name in requested]
    return kernels


def run_kernel(kernel_dir: Path, max_retries: int) -> dict:
    test_script = kernel_dir / "test_kernel.py"
    name = kernel_dir.name
    print(f"\n{'#'*64}")
    print(f"#  Kernel: {name}")
    print(f"{'#'*64}")

    for attempt in range(1, max_retries + 1):
        print(f"\n  ── Attempt {attempt}/{max_retries} ──")
        t0 = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=False,          # let stdout/stderr stream live
            text=True,
        )
        elapsed = time.perf_counter() - t0

        if result.returncode == 0:
            print(f"\n  ✓ {name} PASSED on attempt {attempt}  ({elapsed:.1f}s)")
            return {"kernel": name, "status": "PASS", "attempts": attempt,
                    "elapsed_s": round(elapsed, 2)}

        print(f"\n  ✗ {name} FAILED (exit {result.returncode}) — will retry…")

    # exhausted retries
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fail_record = {"kernel": name, "overall": "FAIL",
                   "note": f"Failed after {max_retries} attempts"}
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(fail_record, indent=2))
    print(f"\n  ✗ {name} FAILED after {max_retries} attempts — skipping")
    return {"kernel": name, "status": "FAIL", "attempts": max_retries}


def print_summary(records):
    print(f"\n{'='*64}")
    print("  FINAL SUMMARY")
    print(f"{'='*64}")
    passed = [r for r in records if r["status"] == "PASS"]
    failed = [r for r in records if r["status"] != "PASS"]
    for r in records:
        icon = "✓" if r["status"] == "PASS" else "✗"
        attempts = r.get("attempts", "?")
        elapsed  = r.get("elapsed_s", "")
        elapsed_str = f"  ({elapsed}s)" if elapsed else ""
        print(f"  {icon}  {r['kernel']:<30}  attempts={attempts}{elapsed_str}")
    print(f"\n  Passed: {len(passed)}/{len(records)}")
    if failed:
        print(f"  Failed: {[r['kernel'] for r in failed]}")
    print(f"{'='*64}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retries",  type=int, default=MAX_RETRIES,
                        help="Max retry attempts per kernel (default 20)")
    parser.add_argument("--kernels",  type=str, default=None,
                        help="Comma-separated list of kernel names to run (all if omitted)")
    args = parser.parse_args()

    kernels = discover_kernels(args.kernels)
    if not kernels:
        print("No kernels found under kernels/*/test_kernel.py")
        sys.exit(1)

    print(f"Found {len(kernels)} kernel(s): {[k.name for k in kernels]}")
    records = []
    for kdir in kernels:
        record = run_kernel(kdir, max_retries=args.retries)
        records.append(record)

    # Write master summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(records, indent=2))
    print_summary(records)

    failed_count = sum(1 for r in records if r["status"] != "PASS")
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()