"""Correctness validation for ViperBench kernel outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch


@dataclass
class TensorComparison:
    """Per-tensor comparison result."""

    name: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    num_mismatched: int
    total_elements: int
    mismatch_fraction: float
    shape_match: bool
    dtype_match: bool


@dataclass
class ValidationResult:
    """Aggregate validation result across all output tensors."""

    passed: bool
    per_tensor: Dict[str, TensorComparison]
    error_message: Optional[str]


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    """Return True for integer tensor dtypes."""
    return dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    )


def _format_mismatched_samples(
    ref: torch.Tensor,
    test: torch.Tensor,
    mismatch_mask: torch.Tensor,
    max_samples: int = 5,
) -> str:
    """Return a human-readable string showing the first *max_samples* mismatched positions."""
    flat_ref = ref.flatten()
    flat_test = test.flatten()
    flat_mask = mismatch_mask.flatten()

    indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    n = min(len(indices), max_samples)
    if n == 0:
        return ""

    lines: List[str] = []
    for i in range(n):
        idx = indices[i].item()
        # Convert flat index back to multi-dimensional index for readability
        nd_idx = []
        remaining = idx
        for dim in reversed(ref.shape):
            nd_idx.append(remaining % dim)
            remaining //= dim
        nd_idx.reverse()
        nd_idx_str = ", ".join(str(x) for x in nd_idx)
        lines.append(
            f"    [{nd_idx_str}]: ref={flat_ref[idx].item():.8g}, test={flat_test[idx].item():.8g}"
        )
    return "\n".join(lines)


def _compare_tensor(
    name: str,
    ref: torch.Tensor,
    test: torch.Tensor,
    atol: float,
    rtol: float,
) -> Tuple[TensorComparison, Optional[str]]:
    """Compare a single pair of tensors and return a comparison result plus optional error detail."""

    total_elements = ref.numel()

    # --- Shape check ---
    if ref.shape != test.shape:
        msg = (
            f"Tensor '{name}': shape mismatch — "
            f"reference {list(ref.shape)} vs test {list(test.shape)}"
        )
        return (
            TensorComparison(
                name=name,
                passed=False,
                max_abs_error=float("inf"),
                max_rel_error=float("inf"),
                mean_abs_error=float("inf"),
                num_mismatched=total_elements,
                total_elements=total_elements,
                mismatch_fraction=1.0,
                shape_match=False,
                dtype_match=(ref.dtype == test.dtype),
            ),
            msg,
        )

    dtype_match = ref.dtype == test.dtype

    # --- Integer output handling ---
    if _is_integer_dtype(ref.dtype) and _is_integer_dtype(test.dtype):
        matched = torch.equal(ref, test)
        if matched:
            return (
                TensorComparison(
                    name=name,
                    passed=True,
                    max_abs_error=0.0,
                    max_rel_error=0.0,
                    mean_abs_error=0.0,
                    num_mismatched=0,
                    total_elements=total_elements,
                    mismatch_fraction=0.0,
                    shape_match=True,
                    dtype_match=dtype_match,
                ),
                None,
            )
        # Compute basic stats for the error message
        diff = (test.to(torch.int64) - ref.to(torch.int64)).abs()
        mismatch_mask = diff != 0
        num_mismatched = int(mismatch_mask.sum().item())
        max_abs = float(diff.max().item())
        mismatch_frac = num_mismatched / max(total_elements, 1)
        samples = _format_mismatched_samples(ref, test, mismatch_mask)
        msg = (
            f"Tensor '{name}': integer exact-match failed — "
            f"{num_mismatched}/{total_elements} elements differ "
            f"({mismatch_frac:.4%}), max abs diff={max_abs}\n"
            f"  First mismatched positions:\n{samples}"
        )
        return (
            TensorComparison(
                name=name,
                passed=False,
                max_abs_error=max_abs,
                max_rel_error=float("inf"),
                mean_abs_error=float(diff.to(torch.float64).mean().item()),
                num_mismatched=num_mismatched,
                total_elements=total_elements,
                mismatch_fraction=mismatch_frac,
                shape_match=True,
                dtype_match=dtype_match,
            ),
            msg,
        )

    # --- Cast to fp32 for numerical comparison ---
    ref_f = ref.to(torch.float32)
    test_f = test.to(torch.float32)

    # --- NaN / Inf check on test output ---
    nan_count = int(torch.isnan(test_f).sum().item())
    inf_count = int(torch.isinf(test_f).sum().item())
    if nan_count > 0 or inf_count > 0:
        parts: List[str] = []
        if nan_count > 0:
            parts.append(f"{nan_count} NaN")
        if inf_count > 0:
            parts.append(f"{inf_count} Inf")
        msg = (
            f"Tensor '{name}': test output contains {', '.join(parts)} "
            f"(out of {total_elements} elements)"
        )
        return (
            TensorComparison(
                name=name,
                passed=False,
                max_abs_error=float("inf"),
                max_rel_error=float("inf"),
                mean_abs_error=float("inf"),
                num_mismatched=nan_count + inf_count,
                total_elements=total_elements,
                mismatch_fraction=(nan_count + inf_count) / max(total_elements, 1),
                shape_match=True,
                dtype_match=dtype_match,
            ),
            msg,
        )

    # --- Numerical comparison ---
    abs_diff = (test_f - ref_f).abs()
    ref_abs = ref_f.abs()

    # rel_diff = abs_diff / max(|ref|, atol)
    rel_denom = torch.clamp(ref_abs, min=atol)
    rel_diff = abs_diff / rel_denom

    # passed = all(abs_diff <= atol + rtol * |ref|)
    mismatch_mask = abs_diff > (atol + rtol * ref_abs)
    num_mismatched = int(mismatch_mask.sum().item())
    mismatch_frac = num_mismatched / max(total_elements, 1)

    max_abs_error = float(abs_diff.max().item())
    max_rel_error = float(rel_diff.max().item())
    mean_abs_error = float(abs_diff.mean().item())

    tensor_passed = num_mismatched == 0

    error_detail: Optional[str] = None
    if not tensor_passed:
        samples = _format_mismatched_samples(ref_f, test_f, mismatch_mask)
        error_detail = (
            f"Tensor '{name}': numerical mismatch — "
            f"{num_mismatched}/{total_elements} elements exceed tolerance "
            f"(atol={atol}, rtol={rtol}), "
            f"mismatch fraction={mismatch_frac:.4%}, "
            f"max abs error={max_abs_error:.6g}, "
            f"max rel error={max_rel_error:.6g}\n"
            f"  First mismatched positions:\n{samples}"
        )

    return (
        TensorComparison(
            name=name,
            passed=tensor_passed,
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
            mean_abs_error=mean_abs_error,
            num_mismatched=num_mismatched,
            total_elements=total_elements,
            mismatch_fraction=mismatch_frac,
            shape_match=True,
            dtype_match=dtype_match,
        ),
        error_detail,
    )


def _normalize_outputs(
    outputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Convert various output formats to a consistent ``{name: tensor}`` dict."""
    if isinstance(outputs, torch.Tensor):
        return {"output_0": outputs}
    if isinstance(outputs, (list, tuple)):
        return {f"output_{i}": t for i, t in enumerate(outputs)}
    if isinstance(outputs, dict):
        return outputs
    raise TypeError(f"Unsupported output type: {type(outputs)}")


def check_correctness(
    reference_outputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    test_outputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> ValidationResult:
    """Compare *test_outputs* against *reference_outputs* and return a :class:`ValidationResult`.

    Parameters
    ----------
    reference_outputs:
        Ground-truth outputs (tensors on any device).
    test_outputs:
        Outputs under test (tensors on any device).
    atol:
        Absolute tolerance for floating-point comparison.
    rtol:
        Relative tolerance for floating-point comparison.
    """

    ref_dict = _normalize_outputs(reference_outputs)
    test_dict = _normalize_outputs(test_outputs)

    # --- Output key check ---
    ref_keys = set(ref_dict.keys())
    test_keys = set(test_dict.keys())

    missing_keys = ref_keys - test_keys
    extra_keys = test_keys - ref_keys

    if missing_keys or extra_keys:
        parts: List[str] = []
        if missing_keys:
            parts.append(f"missing keys in test output: {sorted(missing_keys)}")
        if extra_keys:
            parts.append(f"extra keys in test output: {sorted(extra_keys)}")
        msg = "Output key mismatch — " + "; ".join(parts)
        return ValidationResult(
            passed=False,
            per_tensor={},
            error_message=msg,
        )

    # --- Per-tensor comparison ---
    per_tensor: Dict[str, TensorComparison] = {}
    error_details: List[str] = []
    all_passed = True

    for name in sorted(ref_dict.keys()):
        ref_t = ref_dict[name].detach().cpu()
        test_t = test_dict[name].detach().cpu()
        comparison, detail = _compare_tensor(name, ref_t, test_t, atol, rtol)
        per_tensor[name] = comparison
        if not comparison.passed:
            all_passed = False
            if detail is not None:
                error_details.append(detail)

    error_message: Optional[str] = None
    if not all_passed:
        error_message = "\n\n".join(error_details)

    return ValidationResult(
        passed=all_passed,
        per_tensor=per_tensor,
        error_message=error_message,
    )
