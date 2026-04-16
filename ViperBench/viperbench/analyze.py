"""
ViperBench results analysis — aggregation, tables, and plots.

Usage:
    python -m viperbench.analyze --summary
    python -m viperbench.analyze --category matmul
    python -m viperbench.analyze --roofline matmul --hardware configs/hardware/a100_80gb.json
    python -m viperbench.analyze --compare run_a100/ run_h100/
    python -m viperbench.analyze --latex-tables --output tables/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
KERNELS_DIR = Path(__file__).resolve().parent.parent / "kernels"


def _iter_kernel_dirs() -> List[Tuple[str, Path]]:
    """Yield (kernel_name, kernel_path) for all kernels, supporting grouped subdirs."""
    results = []
    for entry in sorted(KERNELS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "reference.py").exists():
            results.append((entry.name, entry))
        else:
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and (sub / "reference.py").exists():
                    results.append((sub.name, sub))
    return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load all timing.json files from results directory."""
    rdir = results_dir or RESULTS_DIR
    results = []
    if not rdir.exists():
        return results

    for kernel_dir in sorted(rdir.iterdir()):
        timing_path = kernel_dir / "timing.json"
        if timing_path.exists():
            with open(timing_path) as f:
                data = json.load(f)
            results.append(data)
    return results


def load_summary_csv(results_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    """Load summary.csv as list of dicts."""
    rdir = results_dir or RESULTS_DIR
    csv_path = rdir / "summary.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _flatten_results(all_data: List[Dict]) -> List[Dict[str, Any]]:
    """Flatten nested timing.json into flat rows."""
    rows = []
    for kdata in all_data:
        kernel = kdata.get("kernel", "")
        hardware = kdata.get("hardware", "")
        for result_block in kdata.get("results", []):
            config = result_block.get("config", {})
            config_name = config.get("name", "")
            dtype = config.get("dtype", "")
            for impl_name, impl_data in result_block.get("implementations", {}).items():
                row = {
                    "kernel": kernel,
                    "hardware": hardware,
                    "config_name": config_name,
                    "dtype": dtype,
                    "implementation": impl_name,
                }
                row.update(impl_data)
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def generate_summary_table(results_dir: Optional[Path] = None) -> str:
    """Generate per-kernel speedup summary across implementations."""
    all_data = load_all_results(results_dir)
    rows = _flatten_results(all_data)

    # Group by kernel
    by_kernel = defaultdict(list)
    for r in rows:
        by_kernel[r["kernel"]].append(r)

    lines = []
    header = "%-35s  %-8s  %-12s  %-12s  %-12s  %-12s" % (
        "kernel", "dtype", "eager(us)", "compile(us)", "triton(us)", "tilelang(us)")
    lines.append(header)
    lines.append("-" * len(header))

    for kernel in sorted(by_kernel.keys()):
        kernel_rows = by_kernel[kernel]
        # Get median_us per impl, averaged across configs
        impl_medians = defaultdict(list)
        dtypes_seen = set()
        for r in kernel_rows:
            med = r.get("median_us")
            if med:
                try:
                    impl_medians[r["implementation"]].append(float(med))
                except (ValueError, TypeError):
                    pass
            dtypes_seen.add(r.get("dtype", ""))

        def _avg(impl):
            vals = impl_medians.get(impl, [])
            if vals:
                return sum(vals) / len(vals)
            return None

        eager = _avg("pytorch_eager")
        compile_ = _avg("pytorch_compile")
        triton = _avg("triton_impl")
        tilelang = _avg("tilelang_impl")
        dtype_str = ",".join(sorted(dtypes_seen)) if dtypes_seen else ""

        def _fmt(v):
            return "%.1f" % v if v else "-"

        lines.append("%-35s  %-8s  %-12s  %-12s  %-12s  %-12s" % (
            kernel, dtype_str[:8], _fmt(eager), _fmt(compile_),
            _fmt(triton), _fmt(tilelang)))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Category breakdown
# ---------------------------------------------------------------------------

def generate_category_breakdown(category: str,
                                results_dir: Optional[Path] = None) -> str:
    """Per-category detailed breakdown."""
    all_data = load_all_results(results_dir)
    rows = _flatten_results(all_data)

    # Load metadata to filter by category
    cat_kernels = set()
    for kname, kdir in _iter_kernel_dirs():
        meta = kdir / "metadata.json"
        if meta.exists():
            with open(meta) as f:
                m = json.load(f)
            if m.get("category") == category:
                cat_kernels.add(kname)

    filtered = [r for r in rows if r["kernel"] in cat_kernels]
    if not filtered:
        return "No results for category: %s" % category

    lines = ["Category: %s (%d kernels)" % (category, len(cat_kernels)), ""]
    header = "%-30s  %-10s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s" % (
        "kernel", "config", "dtype", "impl", "median_us", "TFLOPS", "SOL%", "bottleneck")
    lines.append(header)
    lines.append("-" * len(header))

    for r in sorted(filtered, key=lambda x: (x["kernel"], x.get("config_name", ""))):
        med = r.get("median_us", "")
        tflops = r.get("achieved_tflops", "")
        sol = r.get("sol_compute_pct") or r.get("sol_memory_pct") or ""
        bn = r.get("bottleneck", "")

        def _f(v):
            if v is None or v == "":
                return "-"
            try:
                return "%.1f" % float(v)
            except (ValueError, TypeError):
                return str(v)

        lines.append("%-30s  %-10s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s" % (
            r["kernel"][:30], str(r.get("config_name", ""))[:10],
            r.get("dtype", ""), r.get("implementation", "")[:10],
            _f(med), _f(tflops), _f(sol), str(bn)[:10]))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coverage matrix
# ---------------------------------------------------------------------------

def generate_coverage_matrix(results_dir: Optional[Path] = None) -> str:
    """Which kernels have implementations in which DSLs."""
    lines = []
    header = "%-35s  %-8s  %-8s  %-8s  %-10s" % (
        "kernel", "eager", "compile", "triton", "tilelang")
    lines.append(header)
    lines.append("-" * len(header))

    for kname, kdir in _iter_kernel_dirs():
        ref = "yes" if (kdir / "reference.py").exists() else "-"
        compile_ = ref  # compile uses reference.py
        triton = "-"
        tilelang = "-"

        tp = kdir / "triton_impl.py"
        if tp.exists():
            triton = "yes"
            try:
                src = tp.read_text()
                if "NOT_IMPLEMENTED = True" in src:
                    triton = "stub"
            except Exception:
                pass

        tlp = kdir / "tilelang_impl.py"
        if tlp.exists():
            tilelang = "yes"
            try:
                src = tlp.read_text()
                if "NOT_IMPLEMENTED = True" in src:
                    tilelang = "stub"
            except Exception:
                pass

        lines.append("%-35s  %-8s  %-8s  %-8s  %-10s" % (
            kname, ref, compile_, triton, tilelang))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Geometric mean speedup
# ---------------------------------------------------------------------------

def compute_geomean_speedups(results_dir: Optional[Path] = None) -> str:
    """Geometric mean speedup of each impl vs PyTorch eager."""
    all_data = load_all_results(results_dir)
    rows = _flatten_results(all_data)

    # Group by (kernel, config_name, dtype) to pair eager with other impls
    groups = defaultdict(dict)
    for r in rows:
        key = (r["kernel"], r.get("config_name", ""), r.get("dtype", ""))
        med = r.get("median_us")
        if med:
            try:
                groups[key][r["implementation"]] = float(med)
            except (ValueError, TypeError):
                pass

    # Compute speedups per impl
    speedups = defaultdict(list)  # type: Dict[str, List[float]]
    for key, impl_times in groups.items():
        eager_t = impl_times.get("pytorch_eager")
        if not eager_t or eager_t <= 0:
            continue
        for impl, t in impl_times.items():
            if impl == "pytorch_eager" or t <= 0:
                continue
            speedups[impl].append(eager_t / t)

    lines = ["Geometric Mean Speedup vs PyTorch Eager", ""]
    for impl in sorted(speedups.keys()):
        vals = speedups[impl]
        if vals:
            log_mean = sum(math.log(v) for v in vals) / len(vals)
            geomean = math.exp(log_mean)
            lines.append("  %-20s  %.3fx  (%d configs)" % (impl, geomean, len(vals)))
        else:
            lines.append("  %-20s  no data" % impl)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ViperBench results analysis")
    parser.add_argument("--summary", action="store_true",
                        help="Generate summary table")
    parser.add_argument("--category", type=str,
                        help="Per-category breakdown")
    parser.add_argument("--coverage", action="store_true",
                        help="Coverage matrix")
    parser.add_argument("--speedups", action="store_true",
                        help="Geometric mean speedups")
    parser.add_argument("--results-dir", type=str,
                        help="Results directory (default: results/)")
    args = parser.parse_args()

    rdir = Path(args.results_dir) if args.results_dir else None

    if args.summary:
        print(generate_summary_table(rdir))
    elif args.category:
        print(generate_category_breakdown(args.category, rdir))
    elif args.coverage:
        print(generate_coverage_matrix(rdir))
    elif args.speedups:
        print(compute_geomean_speedups(rdir))
    else:
        # Default: show summary + speedups
        print(generate_summary_table(rdir))
        print()
        print(compute_geomean_speedups(rdir))


if __name__ == "__main__":
    main()
