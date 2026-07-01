#!/usr/bin/env python3
"""
compute_elib.py -- derive library efficiency (E_lib) from ViperBench profile.csv.

E_lib = (t_lib / t_DSL) x 100%

where t_lib is the PyTorch (cuBLAS / cuDNN) baseline latency and t_DSL is the
Triton or TileLang latency for the SAME (kernel, size). 100% == parity with the
vendor library; below 100% == the DSL kernel is slower.

Paper section 3.4 defines E_lib but no committed script
emitted it (table percentages were hand-derived). This script de-risks
transcription errors by computing every cell from profile.csv directly.

Usage:
    python compute_elib.py [profile.csv] [-o out.csv]

Defaults to ../ViperBench/results/profile.csv relative to this file.
Prints a per-impl summary (geomean + min/max E_lib) and writes a tidy CSV with
columns: kernel,size,impl,pytorch_ms,dsl_ms,elib_percent.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROFILE = os.path.join(HERE, "..", "ViperBench", "results", "profile.csv")
BASELINE = "pytorch"


def load(profile_path):
    """Return {(kernel,size,impl): latency_ms} from profile.csv."""
    lat = {}
    with open(profile_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                lat[(row["kernel"], row["size"], row["impl"])] = float(row["latency_ms"])
            except (KeyError, ValueError):
                continue
    return lat


def compute(lat):
    """Yield (kernel,size,impl,pytorch_ms,dsl_ms,elib_pct) for every DSL impl."""
    kernels_sizes = sorted({(k, s) for (k, s, _i) in lat})
    impls = sorted({i for (_k, _s, i) in lat if i != BASELINE})
    for kernel, size in kernels_sizes:
        base = lat.get((kernel, size, BASELINE))
        if base is None:
            continue
        for impl in impls:
            dsl = lat.get((kernel, size, impl))
            if dsl is None or dsl <= 0:
                continue
            yield kernel, size, impl, base, dsl, base / dsl * 100.0


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", nargs="?", default=DEFAULT_PROFILE,
                    help="path to profile.csv (default: %(default)s)")
    ap.add_argument("-o", "--out", default=None,
                    help="output CSV path (default: <profile_dir>/elib.csv)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.profile):
        sys.exit(f"profile.csv not found: {args.profile}")
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.profile)), "elib.csv")

    rows = list(compute(load(args.profile)))
    if not rows:
        sys.exit("no E_lib rows computed (missing pytorch baseline rows?)")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "size", "impl", "pytorch_ms", "dsl_ms", "elib_percent"])
        for kernel, size, impl, base, dsl, elib in rows:
            w.writerow([kernel, size, impl, f"{base:.6g}", f"{dsl:.6g}", f"{elib:.2f}"])

    # Per-impl summary (geomean across kernels, by size).
    by_impl = defaultdict(list)
    for _k, size, impl, _b, _d, elib in rows:
        by_impl[(impl, size)].append(elib)

    print(f"# E_lib from {os.path.relpath(args.profile)}  ->  {os.path.relpath(out_path)}")
    print(f"# {len(rows)} (kernel,size,impl) cells\n")
    print(f"{'impl':<16}{'size':<8}{'n':>4}{'geomean%':>12}{'min%':>10}{'max%':>10}")
    for (impl, size) in sorted(by_impl):
        v = by_impl[(impl, size)]
        print(f"{impl:<16}{size:<8}{len(v):>4}{geomean(v):>12.1f}{min(v):>10.1f}{max(v):>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
