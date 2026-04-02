"""
JSON output and console reporting for the evaluation system.
Implements EVALUATION_SPEC.md Sections 10 and 11.
"""

import dataclasses
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .config import Status, TestCaseResult


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI color helpers
# ═══════════════════════════════════════════════════════════════════════════════

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_STATUS_COLORS = {
    Status.PASS: _GREEN,
    Status.FAIL_CORRECTNESS: _RED,
    Status.FAIL_RUNTIME_ERROR: _RED,
    Status.OOM: _YELLOW,
    Status.NOT_SUPPORTED_BY_HARDWARE: _CYAN,
    Status.NOT_SUPPORTED_BY_KERNEL: _CYAN,
    Status.NOT_SUPPORTED_BY_REFERENCE: _CYAN,
    Status.TIMEOUT: _YELLOW,
    Status.SKIPPED: _CYAN,
}

_STATUS_ICONS = {
    Status.PASS: "PASS",
    Status.FAIL_CORRECTNESS: "FAIL",
    Status.FAIL_RUNTIME_ERROR: "ERR!",
    Status.OOM: "OOM ",
    Status.NOT_SUPPORTED_BY_HARDWARE: "N/HW",
    Status.NOT_SUPPORTED_BY_KERNEL: "N/KR",
    Status.NOT_SUPPORTED_BY_REFERENCE: "N/RF",
    Status.TIMEOUT: "TIME",
    Status.SKIPPED: "SKIP",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. result_to_dict
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize_float(v: Any) -> Any:
    """Convert non-finite floats to JSON-safe strings."""
    if isinstance(v, float):
        if math.isinf(v):
            return "Inf"
        if math.isnan(v):
            return "NaN"
    return v


def _sanitize_value(v: Any) -> Any:
    """Recursively sanitize a value for JSON serialization."""
    if isinstance(v, float):
        return _sanitize_float(v)
    if isinstance(v, Status):
        return v.value
    if isinstance(v, list):
        return [_sanitize_value(item) for item in v]
    if isinstance(v, dict):
        return {k: _sanitize_value(val) for k, val in v.items()}
    return v


def result_to_dict(result: TestCaseResult) -> dict:
    """Convert a TestCaseResult dataclass to a JSON-serializable dict.

    - Status enum is converted to its string value.
    - None values stay as None (JSON null).
    - float('inf') / float('nan') become "Inf" / "NaN".
    """
    d = dataclasses.asdict(result)
    return {k: _sanitize_value(v) for k, v in d.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. validate_result
# ═══════════════════════════════════════════════════════════════════════════════

_REQUIRED_FIELDS = [
    "op_name",
    "backend",
    "reference_backend",
    "device",
    "dtype_in",
    "dtype_out",
    "shape",
    "layout",
    "warmup_runs",
    "timed_runs",
    "status",
]

_LATENCY_FIELDS = [
    "latency_ms_mean",
    "latency_ms_median",
    "latency_ms_std",
    "latency_ms_p95",
    "latency_ms_min",
]

_CORRECTNESS_FIELDS = [
    "max_abs_error",
    "mean_abs_error",
    "max_rel_error",
    "bad_elem_ratio",
    "correctness_pass",
]

_ALLOWED_STATUSES = {s.value for s in Status}


def validate_result(result_dict: dict) -> Tuple[bool, List[str]]:
    """Validate a result dict per Section 11 rules.

    Returns (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    # Check required fields exist
    for field in _REQUIRED_FIELDS:
        if field not in result_dict:
            errors.append(f"Missing required field: {field}")

    # Check status is allowed
    status = result_dict.get("status")
    if status is not None and status not in _ALLOWED_STATUSES:
        errors.append(f"Invalid status value: {status!r}")

    # If PASS, latency and correctness metrics must be non-None
    if status == Status.PASS.value:
        for field in _LATENCY_FIELDS:
            val = result_dict.get(field)
            if val is None:
                errors.append(
                    f"Status is PASS but {field} is None"
                )

        for field in _CORRECTNESS_FIELDS:
            val = result_dict.get(field)
            if val is None:
                errors.append(
                    f"Status is PASS but {field} is None"
                )

    return (len(errors) == 0, errors)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. save_results_json
# ═══════════════════════════════════════════════════════════════════════════════

def _count_by_status(results: List[TestCaseResult]) -> Dict[str, int]:
    """Count results grouped by status."""
    counts: Dict[str, int] = defaultdict(int)
    for r in results:
        counts[r.status.value] += 1
    return dict(counts)


def save_results_json(
    results: List[TestCaseResult],
    env: dict,
    output_path: str,
) -> None:
    """Write JSON report file with environment, summary, and results.

    Creates parent directories if needed. Validates each result before writing.
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    status_counts = _count_by_status(results)

    summary = {
        "total": len(results),
        "pass": status_counts.get(Status.PASS.value, 0),
        "fail_correctness": status_counts.get(
            Status.FAIL_CORRECTNESS.value, 0
        ),
        "fail_runtime_error": status_counts.get(
            Status.FAIL_RUNTIME_ERROR.value, 0
        ),
        "oom": status_counts.get(Status.OOM.value, 0),
        "not_supported": (
            status_counts.get(Status.NOT_SUPPORTED_BY_HARDWARE.value, 0)
            + status_counts.get(Status.NOT_SUPPORTED_BY_KERNEL.value, 0)
            + status_counts.get(Status.NOT_SUPPORTED_BY_REFERENCE.value, 0)
        ),
        "timeout": status_counts.get(Status.TIMEOUT.value, 0),
        "skipped": status_counts.get(Status.SKIPPED.value, 0),
    }

    result_dicts = []
    validation_warnings: List[str] = []
    for i, r in enumerate(results):
        rd = result_to_dict(r)
        valid, errs = validate_result(rd)
        if not valid:
            for e in errs:
                validation_warnings.append(f"Result[{i}] ({r.op_name}): {e}")
        result_dicts.append(rd)

    output = {
        "environment": env,
        "summary": summary,
        "results": result_dicts,
    }

    if validation_warnings:
        output["validation_warnings"] = validation_warnings

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. print_console_summary
# ═══════════════════════════════════════════════════════════════════════════════

def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  -  "
    return f"{100.0 * n / total:5.1f}%"


def print_console_summary(results: List[TestCaseResult]) -> None:
    """Print a formatted summary table to the console."""
    total = len(results)
    if total == 0:
        print(f"{_BOLD}No results to report.{_RESET}")
        return

    status_counts = _count_by_status(results)
    n_pass = status_counts.get(Status.PASS.value, 0)
    n_fail = (
        status_counts.get(Status.FAIL_CORRECTNESS.value, 0)
        + status_counts.get(Status.FAIL_RUNTIME_ERROR.value, 0)
    )

    # Overall summary
    print()
    print(f"{_BOLD}{'=' * 72}{_RESET}")
    print(f"{_BOLD}  EVALUATION SUMMARY{_RESET}")
    print(f"{_BOLD}{'=' * 72}{_RESET}")
    color = _GREEN if n_fail == 0 else _RED
    print(f"  Total: {total}   "
          f"{_GREEN}Pass: {n_pass}{_RESET}   "
          f"{color}Fail: {n_fail}{_RESET}")
    print()

    # Per-status breakdown
    print(f"  {_BOLD}{'Status':<30} {'Count':>6} {'Pct':>7}{_RESET}")
    print(f"  {'-' * 45}")
    for s in Status:
        cnt = status_counts.get(s.value, 0)
        if cnt == 0:
            continue
        c = _STATUS_COLORS.get(s, _RESET)
        print(f"  {c}{s.value:<30}{_RESET} {cnt:>6} {_pct(cnt, total):>7}")
    print()

    # Per-kernel breakdown
    kernel_results: Dict[str, List[TestCaseResult]] = defaultdict(list)
    for r in results:
        kernel_results[r.op_name].append(r)

    print(f"  {_BOLD}{'Kernel':<32} {'Pass':>8} "
          f"{'Best ms':>9} {'Status':>6}{_RESET}")
    print(f"  {'-' * 60}")
    for kname in sorted(kernel_results.keys()):
        kresults = kernel_results[kname]
        k_total = len(kresults)
        k_pass = sum(1 for r in kresults if r.status == Status.PASS)
        # Best latency among passing results
        passing_latencies = [
            r.latency_ms_min
            for r in kresults
            if r.status == Status.PASS and r.latency_ms_min is not None
        ]
        best_lat = f"{min(passing_latencies):.3f}" if passing_latencies else "-"
        # Worst status icon
        statuses = {r.status for r in kresults}
        if Status.FAIL_CORRECTNESS in statuses or Status.FAIL_RUNTIME_ERROR in statuses:
            icon = f"{_RED}FAIL{_RESET}"
        elif Status.OOM in statuses or Status.TIMEOUT in statuses:
            icon = f"{_YELLOW}WARN{_RESET}"
        elif k_pass == k_total:
            icon = f"{_GREEN}OK  {_RESET}"
        else:
            icon = f"{_CYAN}SKIP{_RESET}"

        name_trunc = kname[:32]
        pass_str = f"{k_pass}/{k_total}"
        print(f"  {name_trunc:<32} {pass_str:>8} "
              f"{best_lat:>9} {icon}")
    print()

    # Per-dtype breakdown
    dtype_results: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for r in results:
        dtype_results[r.dtype_in][r.status.value] += 1

    if dtype_results:
        print(f"  {_BOLD}{'Dtype':<16} {'Pass':>6} {'Fail':>6} "
              f"{'Other':>6} {'Total':>6}{_RESET}")
        print(f"  {'-' * 44}")
        for dt in sorted(dtype_results.keys()):
            counts = dtype_results[dt]
            dt_total = sum(counts.values())
            dt_pass = counts.get(Status.PASS.value, 0)
            dt_fail = (
                counts.get(Status.FAIL_CORRECTNESS.value, 0)
                + counts.get(Status.FAIL_RUNTIME_ERROR.value, 0)
            )
            dt_other = dt_total - dt_pass - dt_fail
            print(f"  {dt:<16} {dt_pass:>6} {dt_fail:>6} "
                  f"{dt_other:>6} {dt_total:>6}")
        print()

    # Per-size-bucket breakdown (derived from input_bytes_total)
    bucket_results: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for r in results:
        bucket = _bytes_to_bucket(r.input_bytes_total + r.output_bytes_total)
        bucket_results[bucket][r.status.value] += 1

    if bucket_results:
        print(f"  {_BOLD}{'Size Bucket':<16} {'Pass':>6} {'Fail':>6} "
              f"{'Other':>6} {'Total':>6}{_RESET}")
        print(f"  {'-' * 44}")
        for bkt in sorted(bucket_results.keys()):
            counts = bucket_results[bkt]
            b_total = sum(counts.values())
            b_pass = counts.get(Status.PASS.value, 0)
            b_fail = (
                counts.get(Status.FAIL_CORRECTNESS.value, 0)
                + counts.get(Status.FAIL_RUNTIME_ERROR.value, 0)
            )
            b_other = b_total - b_pass - b_fail
            print(f"  {bkt:<16} {b_pass:>6} {b_fail:>6} "
                  f"{b_other:>6} {b_total:>6}")
        print()

    print(f"{_BOLD}{'=' * 72}{_RESET}")


def _bytes_to_bucket(nbytes: int) -> str:
    """Map total bytes to a human-readable size bucket label."""
    gb = nbytes / (1024 ** 3)
    if gb < 1:
        return "<1GB"
    elif gb < 3:
        return "2GB"
    elif gb < 6:
        return "4GB"
    elif gb < 12:
        return "8GB"
    elif gb < 24:
        return "16GB"
    elif gb < 48:
        return "32GB"
    else:
        return "64GB"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. print_kernel_detail
# ═══════════════════════════════════════════════════════════════════════════════

def print_kernel_detail(
    kernel: str,
    results: List[TestCaseResult],
) -> None:
    """Print detailed per-test-case output for a single kernel."""
    kernel_results = [r for r in results if r.op_name == kernel]
    if not kernel_results:
        print(f"No results found for kernel: {kernel}")
        return

    print()
    print(f"{_BOLD}Kernel: {kernel}{_RESET}")
    print(f"{'-' * 72}")

    for i, r in enumerate(kernel_results):
        sc = _STATUS_COLORS.get(r.status, _RESET)
        si = _STATUS_ICONS.get(r.status, "????")
        print(f"  [{i}] {sc}{si}{_RESET}  dtype={r.dtype_in} "
              f"shape={r.shape} layout={r.layout}")

        if r.status == Status.PASS or r.latency_ms_mean is not None:
            print(f"      Latency (ms): mean={_fmt(r.latency_ms_mean)} "
                  f"median={_fmt(r.latency_ms_median)} "
                  f"std={_fmt(r.latency_ms_std)} "
                  f"p95={_fmt(r.latency_ms_p95)} "
                  f"min={_fmt(r.latency_ms_min)}")
        if r.bandwidth_gbps is not None:
            print(f"      Bandwidth: {r.bandwidth_gbps:.2f} GB/s")
        if r.max_abs_error is not None:
            print(f"      Correctness: max_abs={_fmt(r.max_abs_error)} "
                  f"mean_abs={_fmt(r.mean_abs_error)} "
                  f"max_rel={_fmt(r.max_rel_error)} "
                  f"bad_ratio={_fmt(r.bad_elem_ratio)}")
        if r.notes:
            print(f"      Notes: {r.notes}")
    print()


def _fmt(v: Optional[float]) -> str:
    """Format an optional float for display."""
    if v is None:
        return "-"
    if math.isinf(v):
        return "Inf"
    if math.isnan(v):
        return "NaN"
    if abs(v) < 1e-6:
        return f"{v:.2e}"
    return f"{v:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. generate_summary_stats
# ═══════════════════════════════════════════════════════════════════════════════

def _make_status_dict() -> Dict[str, int]:
    return {s.value: 0 for s in Status}


def generate_summary_stats(results: List[TestCaseResult]) -> dict:
    """Aggregate statistics across all results.

    Returns dict with keys:
      - total, per-status counts
      - per_category: {category: {status: count, ...}}
      - per_dtype: {dtype: {status: count, ...}}
      - per_size_bucket: {bucket: {status: count, ...}}
    """
    total = len(results)
    overall = _make_status_dict()
    per_category: Dict[str, Dict[str, int]] = defaultdict(_make_status_dict)
    per_dtype: Dict[str, Dict[str, int]] = defaultdict(_make_status_dict)
    per_size_bucket: Dict[str, Dict[str, int]] = defaultdict(_make_status_dict)

    for r in results:
        sv = r.status.value
        overall[sv] += 1

        # Category is derived from op_name: take the part before the first "/"
        # or use the whole name if no "/" exists
        category = r.op_name.split("/")[0] if "/" in r.op_name else r.op_name
        per_category[category][sv] += 1

        per_dtype[r.dtype_in][sv] += 1

        bucket = _bytes_to_bucket(r.input_bytes_total + r.output_bytes_total)
        per_size_bucket[bucket][sv] += 1

    return {
        "total": total,
        **overall,
        "per_category": dict(per_category),
        "per_dtype": dict(per_dtype),
        "per_size_bucket": dict(per_size_bucket),
    }
