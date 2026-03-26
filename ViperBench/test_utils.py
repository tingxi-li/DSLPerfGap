"""
Shared test utilities for ViperBench kernel verification (PyTorch vs Triton).
"""
import sys
import json
import time
import traceback
from pathlib import Path
import torch

# ── tolerances by dtype ──────────────────────────────────────────────────────
TOLERANCES = {
    torch.float32:  dict(atol=1e-5,  rtol=1e-5),
    torch.float16:  dict(atol=1e-3,  rtol=1e-3),
    torch.bfloat16: dict(atol=1e-2,  rtol=1e-2),
}

RESULTS_DIR = Path(__file__).parent / "results"


def get_tol(dtype, loose=False):
    tol = TOLERANCES.get(dtype, dict(atol=1e-5, rtol=1e-5))
    if loose:
        tol = {k: v * 2 for k, v in tol.items()}
    return tol


def compare_tensors(ref, test, dtype, loose=False):
    """Compare two tensors. Returns (passed: bool, max_err: float)."""
    tol = get_tol(dtype, loose=loose)
    ref_f = ref.float()
    test_f = test.float()
    max_err = (ref_f - test_f).abs().max().item()
    passed = torch.allclose(ref_f, test_f, **tol)
    return passed, max_err


def compare_tuple_outputs(ref_tuple, test_tuple, dtype, loose=False):
    """Compare tuples of tensors. Returns (passed: bool, max_err: float)."""
    overall_pass = True
    overall_max_err = 0.0
    for ref, test in zip(ref_tuple, test_tuple):
        if isinstance(ref, torch.Tensor) and isinstance(test, torch.Tensor):
            passed, max_err = compare_tensors(ref, test, dtype, loose=loose)
            overall_pass = overall_pass and passed
            overall_max_err = max(overall_max_err, max_err)
    return overall_pass, overall_max_err


def run_test(kernel_name, test_cases, pytorch_fn, triton_fn,
             loose_tol=False, compare_fn=None):
    """
    Run test cases comparing PyTorch and Triton implementations.

    kernel_name : str
    test_cases  : list of dicts, each with:
                  - "name": str label
                  - "inputs": dict or tuple of torch.Tensors to pass to both fns
                  - "dtype": torch.dtype for tolerance selection
    pytorch_fn  : callable(*inputs) -> Tensor or tuple of Tensors
    triton_fn   : callable(*inputs) -> Tensor or tuple of Tensors
    loose_tol   : use 2x tolerances (for reductions)
    compare_fn  : optional custom comparison fn(ref, test, dtype, loose) -> (bool, float)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_pass = True
    records = []

    print(f"\n{'='*60}")
    print(f"  Kernel: {kernel_name}")
    print(f"{'='*60}")

    for tc in test_cases:
        label = tc["name"]
        dtype = tc["dtype"]
        inputs = tc["inputs"]
        try:
            # Normalize inputs to a list for unpacking
            if isinstance(inputs, dict):
                ref_out = pytorch_fn(**inputs)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                tri_out = triton_fn(**inputs)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000
            elif isinstance(inputs, (list, tuple)):
                ref_out = pytorch_fn(*inputs)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                tri_out = triton_fn(*inputs)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000
            else:
                ref_out = pytorch_fn(inputs)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                tri_out = triton_fn(inputs)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000

            # Compare
            cmp = compare_fn or _default_compare
            passed, max_err = cmp(ref_out, tri_out, dtype, loose_tol)

            if not passed:
                print(f"  FAIL  {label}  |  max_err={max_err:.2e}  tol={get_tol(dtype, loose_tol)}")
                all_pass = False
                records.append({"name": label, "status": "FAIL", "max_err": max_err})
            else:
                print(f"  PASS  {label}  |  max_err={max_err:.2e}  time={elapsed_ms:.2f}ms")
                records.append({"name": label, "status": "PASS",
                                "max_err": max_err, "time_ms": elapsed_ms})

        except Exception:
            tb = traceback.format_exc()
            print(f"  ERROR {label}\n{tb}")
            all_pass = False
            records.append({"name": label, "status": "ERROR", "traceback": tb})

    # Write JSON result
    result = {
        "kernel": kernel_name,
        "overall": "PASS" if all_pass else "FAIL",
        "test_cases": records,
    }
    out_path = RESULTS_DIR / f"{kernel_name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}  -> {out_path}")
    print(f"{'='*60}\n")

    sys.exit(0 if all_pass else 1)


def run_tilelang_test(kernel_name, test_cases, pytorch_fn, tilelang_fn,
                      loose_tol=False, compare_fn=None):
    """
    Run test cases comparing PyTorch and TileLang implementations.
    Same interface as run_test but for tilelang validation.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_pass = True
    records = []

    print(f"\n{'='*60}")
    print(f"  Kernel (TileLang): {kernel_name}")
    print(f"{'='*60}")

    for tc in test_cases:
        label = tc["name"]
        dtype = tc["dtype"]
        inputs = tc["inputs"]
        try:
            if isinstance(inputs, dict):
                ref_out = pytorch_fn(**inputs)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                tl_out = tilelang_fn(**inputs)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000
            elif isinstance(inputs, (list, tuple)):
                ref_out = pytorch_fn(*inputs)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                tl_out = tilelang_fn(*inputs)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000
            else:
                ref_out = pytorch_fn(inputs)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                tl_out = tilelang_fn(inputs)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000

            cmp = compare_fn or _default_compare
            passed, max_err = cmp(ref_out, tl_out, dtype, loose_tol)

            if not passed:
                print(f"  FAIL  {label}  |  max_err={max_err:.2e}  tol={get_tol(dtype, loose_tol)}")
                all_pass = False
                records.append({"name": label, "status": "FAIL", "max_err": max_err})
            else:
                print(f"  PASS  {label}  |  max_err={max_err:.2e}  time={elapsed_ms:.2f}ms")
                records.append({"name": label, "status": "PASS",
                                "max_err": max_err, "time_ms": elapsed_ms})

        except Exception:
            tb = traceback.format_exc()
            print(f"  ERROR {label}\n{tb}")
            all_pass = False
            records.append({"name": label, "status": "ERROR", "traceback": tb})

    result = {
        "kernel": kernel_name + "_tilelang",
        "overall": "PASS" if all_pass else "FAIL",
        "test_cases": records,
    }
    out_path = RESULTS_DIR / f"{kernel_name}_tilelang.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}  -> {out_path}")
    print(f"{'='*60}\n")

    sys.exit(0 if all_pass else 1)


def _default_compare(ref_out, tri_out, dtype, loose):
    """Default comparison: handles single tensor or tuple of tensors."""
    if isinstance(ref_out, (tuple, list)) and isinstance(tri_out, (tuple, list)):
        return compare_tuple_outputs(ref_out, tri_out, dtype, loose)
    elif isinstance(ref_out, torch.Tensor) and isinstance(tri_out, torch.Tensor):
        return compare_tensors(ref_out, tri_out, dtype, loose)
    else:
        # Try converting to tensor
        return compare_tensors(torch.tensor(ref_out), torch.tensor(tri_out), dtype, loose)
