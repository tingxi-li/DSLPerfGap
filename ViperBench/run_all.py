#!/usr/bin/env python3
"""Run all ViperBench kernel tests and print a summary."""
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent


def main():
    kernel_dirs = sorted(
        d for d in BENCH_DIR.iterdir()
        if d.is_dir() and (d / "test.py").exists()
    )

    results = {}
    for kdir in kernel_dirs:
        name = kdir.name
        print(f"\n>>> Running {name} ...")
        ret = subprocess.run(
            [sys.executable, str(kdir / "test.py")],
            capture_output=False,
        )
        results[name] = "PASS" if ret.returncode == 0 else "FAIL"

    # Summary
    print(f"\n{'='*60}")
    print("  ViperBench Summary")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    for name, status in results.items():
        marker = "PASS" if status == "PASS" else "FAIL"
        print(f"  {marker}  {name}")
    print(f"\n  {passed}/{total} kernels passed")
    print(f"{'='*60}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
