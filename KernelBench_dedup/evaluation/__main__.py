"""
CLI entry point for the evaluation system.

Usage:
    python -m KernelBench_dedup.evaluation [options]
    python -m evaluation [options]  (when cwd is KernelBench_dedup/)
"""

import argparse
import sys

import torch

from .config import (
    DEFAULT_DTYPES,
    DEFAULT_LAYOUTS,
    DEFAULT_SHAPE_FAMILIES,
    DEFAULT_SIZE_BUCKETS,
    DEFAULT_VALUE_DISTS,
    TIMEOUT_SECONDS,
    TIMED_RUNS,
    WARMUP_RUNS,
    Layout,
    ShapeFamily,
    ValueDist,
)
from .reporter import print_console_summary, save_results_json
from .runner import collect_environment, discover_kernels, run_all


# ═══════════════════════════════════════════════════════════════════════════════
# Enum / type parsers
# ═══════════════════════════════════════════════════════════════════════════════

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}

_SHAPE_FAMILY_MAP = {sf.value: sf for sf in ShapeFamily}
_LAYOUT_MAP = {lo.value: lo for lo in Layout}
_VALUE_DIST_MAP = {vd.value: vd for vd in ValueDist}


def _parse_csv(s: str) -> list:
    """Split comma-separated string, stripping whitespace."""
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_dtypes(csv_str: str) -> list:
    names = _parse_csv(csv_str)
    result = []
    for n in names:
        if n not in _DTYPE_MAP:
            print(f"Warning: unknown dtype '{n}', skipping. "
                  f"Valid: {list(_DTYPE_MAP.keys())}")
            continue
        result.append(_DTYPE_MAP[n])
    return result


def _parse_shape_families(csv_str: str) -> list:
    names = _parse_csv(csv_str)
    result = []
    for n in names:
        if n not in _SHAPE_FAMILY_MAP:
            print(f"Warning: unknown shape family '{n}', skipping. "
                  f"Valid: {list(_SHAPE_FAMILY_MAP.keys())}")
            continue
        result.append(_SHAPE_FAMILY_MAP[n])
    return result


def _parse_layouts(csv_str: str) -> list:
    names = _parse_csv(csv_str)
    result = []
    for n in names:
        if n not in _LAYOUT_MAP:
            print(f"Warning: unknown layout '{n}', skipping. "
                  f"Valid: {list(_LAYOUT_MAP.keys())}")
            continue
        result.append(_LAYOUT_MAP[n])
    return result


def _parse_value_dists(csv_str: str) -> list:
    names = _parse_csv(csv_str)
    result = []
    for n in names:
        if n not in _VALUE_DIST_MAP:
            print(f"Warning: unknown value dist '{n}', skipping. "
                  f"Valid: {list(_VALUE_DIST_MAP.keys())}")
            continue
        result.append(_VALUE_DIST_MAP[n])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive KernelBench evaluation harness"
    )
    parser.add_argument(
        "--level",
        choices=["level1", "level2", "level3", "tritonbench"],
        help="Run only kernels from this level",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        help="Filter kernels by substring match on name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_full.json",
        help="JSON output path (default: results/eval_full.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_SECONDS,
        help=f"Per-kernel timeout in seconds (default: {TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--size-buckets",
        type=str,
        default=",".join(DEFAULT_SIZE_BUCKETS),
        help="Comma-separated size buckets (default: <1GB,2GB,4GB)",
    )
    parser.add_argument(
        "--shape-families",
        type=str,
        default=",".join(sf.value for sf in DEFAULT_SHAPE_FAMILIES),
        help="Comma-separated shape families (default: 1d_flat,2d_square,4d_nchw)",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        default="float32,float16",
        help="Comma-separated dtypes (default: float32,float16)",
    )
    parser.add_argument(
        "--layouts",
        type=str,
        default=",".join(lo.value for lo in DEFAULT_LAYOUTS),
        help="Comma-separated layouts (default: contiguous)",
    )
    parser.add_argument(
        "--value-dists",
        type=str,
        default=",".join(vd.value for vd in DEFAULT_VALUE_DISTS),
        help="Comma-separated value distributions (default: uniform)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_RUNS,
        help=f"Warmup iterations (default: {WARMUP_RUNS})",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=TIMED_RUNS,
        help=f"Timed iterations (default: {TIMED_RUNS})",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Use minimal acceptance defaults (3 size buckets, 3 shapes, 2 dtypes)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Just list discovered kernels and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-case output for each kernel",
    )

    args = parser.parse_args()

    # Resolve levels
    levels = [args.level] if args.level else None

    # Handle --list
    if args.list:
        kernels = discover_kernels(levels=levels, kernel_filter=args.kernel)
        for level, name, kdir in kernels:
            print(f"  {level:12s}  {name}")
        print(f"\nTotal: {len(kernels)} kernels")
        return

    # Parse enum arguments
    if args.minimal:
        size_buckets = list(DEFAULT_SIZE_BUCKETS)
        shape_families = list(DEFAULT_SHAPE_FAMILIES)
        dtypes = list(DEFAULT_DTYPES)
        layouts = list(DEFAULT_LAYOUTS)
        value_dists = list(DEFAULT_VALUE_DISTS)
    else:
        size_buckets = _parse_csv(args.size_buckets)
        shape_families = _parse_shape_families(args.shape_families)
        dtypes = _parse_dtypes(args.dtypes)
        layouts = _parse_layouts(args.layouts)
        value_dists = _parse_value_dists(args.value_dists)

    # Collect environment
    env = collect_environment()

    print(f"Environment: {env['device']} | PyTorch {env['pytorch_version']} | "
          f"CUDA {env['driver_version']}")
    print(f"Config: sizes={size_buckets} shapes={[sf.value for sf in shape_families]} "
          f"dtypes={[str(d).replace('torch.','') for d in dtypes]} "
          f"layouts={[lo.value for lo in layouts]} "
          f"value_dists={[vd.value for vd in value_dists]}")
    print(f"Benchmark: warmup={args.warmup} timed={args.timed_runs} timeout={args.timeout}s")
    print(f"{'=' * 72}")

    # Run
    results = run_all(
        levels=levels,
        kernel_filter=args.kernel,
        size_buckets=size_buckets,
        shape_families=shape_families,
        dtypes=dtypes,
        layouts=layouts,
        value_dists=value_dists,
        env=env,
        timeout=args.timeout,
        warmup=args.warmup,
        timed=args.timed_runs,
        verbose=args.verbose,
    )

    # Summary
    print_console_summary(results)

    # Save
    save_results_json(results, env, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
