"""
Correctness evaluation module.

Implements per-element error metrics, correctness pass/fail logic,
and determinism checking per EVALUATION_SPEC.md Sections 2.3, 2.4, 4.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .config import TOLERANCES, EPS, DETERMINISM_RUNS


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def flatten_outputs(output: Any) -> List[torch.Tensor]:
    """Convert any output type (Tensor, tuple, list, None) to a flat list of tensors.

    Non-tensor elements (e.g. None, int, float) are silently skipped.
    """
    if output is None:
        return []
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        tensors: List[torch.Tensor] = []
        for item in output:
            tensors.extend(flatten_outputs(item))
        return tensors
    # Non-tensor scalar — skip
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Per-element error
# ═══════════════════════════════════════════════════════════════════════════════


def compute_element_error(
    output: torch.Tensor,
    reference: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Compute per-element error: e_i = max(|y_i - y_ref_i|, |y_i - y_ref_i| / (|y_ref_i| + eps)).

    Both tensors are cast to float64, flattened, and the result is a float64
    tensor of per-element errors.
    """
    y = output.detach().flatten().to(torch.float64)
    y_ref = reference.detach().flatten().to(torch.float64)

    abs_diff = torch.abs(y - y_ref)
    rel_diff = abs_diff / (torch.abs(y_ref) + eps)

    return torch.max(abs_diff, rel_diff)


# ═══════════════════════════════════════════════════════════════════════════════
# Correctness metrics
# ═══════════════════════════════════════════════════════════════════════════════


def compute_correctness_metrics(
    output: Any,
    reference: Any,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Compute correctness metrics comparing *output* to *reference*.

    Handles single tensors, tuples/lists of tensors (element-wise comparison),
    and integer dtypes (exact match).

    Returns a dict with keys:
        max_abs_error, mean_abs_error, max_rel_error,
        bad_elem_ratio, correctness_pass
    """
    out_tensors = flatten_outputs(output)
    ref_tensors = flatten_outputs(reference)

    # Mismatched number of output tensors
    if len(out_tensors) != len(ref_tensors):
        return _fail_metrics(note="output/reference tensor count mismatch")

    # No tensors to compare
    if len(out_tensors) == 0:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_rel_error": 0.0,
            "bad_elem_ratio": 0.0,
            "correctness_pass": True,
        }

    # Look up tolerances; fall back to float32 for unknown dtypes
    tau_elem, tau_ratio = TOLERANCES.get(dtype, TOLERANCES[torch.float32])

    # Integer dtypes use exact match
    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
        tau_elem = 0.0
        tau_ratio = 0.0

    all_errors: List[torch.Tensor] = []
    all_abs_diffs: List[torch.Tensor] = []
    all_rel_diffs: List[torch.Tensor] = []

    for y, y_ref in zip(out_tensors, ref_tensors):
        # Shape mismatch
        if y.shape != y_ref.shape:
            return _fail_metrics(note="shape mismatch")

        y64 = y.detach().flatten().to(torch.float64)
        y_ref64 = y_ref.detach().flatten().to(torch.float64)

        abs_diff = torch.abs(y64 - y_ref64)
        rel_diff = abs_diff / (torch.abs(y_ref64) + EPS)
        elem_err = torch.max(abs_diff, rel_diff)

        all_errors.append(elem_err)
        all_abs_diffs.append(abs_diff)
        all_rel_diffs.append(rel_diff)

    errors = torch.cat(all_errors)
    abs_diffs = torch.cat(all_abs_diffs)
    rel_diffs = torch.cat(all_rel_diffs)

    # NaN check
    if torch.isnan(errors).any():
        return _fail_metrics(note="NaN detected in output")

    max_abs_error = float(abs_diffs.max())
    mean_abs_error = float(abs_diffs.mean())
    max_rel_error = float(rel_diffs.max())

    total_elems = errors.numel()
    bad_count = int((errors > tau_elem).sum())
    bad_elem_ratio = bad_count / total_elems if total_elems > 0 else 0.0

    correctness_pass = bad_elem_ratio <= tau_ratio

    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
        "bad_elem_ratio": bad_elem_ratio,
        "correctness_pass": correctness_pass,
    }


def _fail_metrics(note: str = "") -> Dict[str, Any]:
    """Return a metrics dict indicating failure with infinite errors."""
    return {
        "max_abs_error": float("inf"),
        "mean_abs_error": float("inf"),
        "max_rel_error": float("inf"),
        "bad_elem_ratio": 1.0,
        "correctness_pass": False,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism check
# ═══════════════════════════════════════════════════════════════════════════════


def check_determinism(
    run_fn: Callable,
    inputs: Any,
    k: int = DETERMINISM_RUNS,
) -> Dict[str, Any]:
    """Run *run_fn(inputs)* K times and check whether outputs are bit-identical.

    Returns a dict with keys:
        is_deterministic, reference_variance, reference_run_count
    """
    collected: List[List[torch.Tensor]] = []

    for _ in range(k):
        raw_output = run_fn(inputs)
        tensors = flatten_outputs(raw_output)
        # Detach and move to CPU for comparison
        collected.append([t.detach().cpu() for t in tensors])

    if len(collected) == 0 or len(collected[0]) == 0:
        return {
            "is_deterministic": True,
            "reference_variance": 0.0,
            "reference_run_count": k,
        }

    max_variance = 0.0
    num_outputs = len(collected[0])

    for t_idx in range(num_outputs):
        # Stack the t_idx-th tensor across all K runs
        stacked = torch.stack(
            [collected[run_idx][t_idx].to(torch.float64) for run_idx in range(k)],
            dim=0,
        )
        variance = torch.var(stacked, dim=0)
        run_max_var = float(variance.max())
        if run_max_var > max_variance:
            max_variance = run_max_var

    # Treat as deterministic if variance is within machine epsilon for float64
    is_deterministic = max_variance <= torch.finfo(torch.float64).eps

    return {
        "is_deterministic": is_deterministic,
        "reference_variance": max_variance,
        "reference_run_count": k,
    }
