"""
Universal test harness for TileLang kernel verification.
Usage: python test_kernel.py
Exit 0 = all shapes pass, Exit 1 = any failure.
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

def get_tol(dtype, loose=False):
    tol = TOLERANCES.get(dtype, dict(atol=1e-5, rtol=1e-5))
    if loose:
        tol = {k: v * 2 for k, v in tol.items()}
    return tol


def run_test(kernel_name, shapes, pytorch_fn, tilelang_fn,
             make_inputs, loose_tol=False, results_dir="tests/results"):
    """
    kernel_name  : str — used for logging
    shapes       : list of shape-dicts passed to make_inputs
    pytorch_fn   : callable(inputs) -> torch.Tensor
    tilelang_fn  : callable(inputs) -> torch.Tensor
    make_inputs  : callable(shape_dict) -> dict of tensors
    loose_tol    : bool — use 2× tolerances (reductions)
    results_dir  : where to write JSON result log
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    all_pass = True
    records = []

    print(f"\n{'='*60}")
    print(f"  Kernel: {kernel_name}")
    print(f"{'='*60}")

    for i, shape in enumerate(shapes):
        label = f"shape[{i}] {shape}"
        try:
            inputs = make_inputs(shape)
            dtype  = next(iter(inputs.values())).dtype
            tol    = get_tol(dtype, loose=loose_tol)

            # Reference
            ref_out = pytorch_fn(inputs)
            torch.cuda.synchronize()

            # TileLang
            t0 = time.perf_counter()
            tl_out = tilelang_fn(inputs)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Compare
            if not torch.allclose(ref_out.float(), tl_out.float(), **tol):
                max_err = (ref_out.float() - tl_out.float()).abs().max().item()
                print(f"  FAIL  {label}  |  max_err={max_err:.2e}  tol={tol}")
                all_pass = False
                records.append({"shape": str(shape), "status": "FAIL",
                                 "max_err": max_err})
            else:
                max_err = (ref_out.float() - tl_out.float()).abs().max().item()
                print(f"  PASS  {label}  |  max_err={max_err:.2e}  time={elapsed_ms:.2f}ms")
                records.append({"shape": str(shape), "status": "PASS",
                                 "max_err": max_err, "time_ms": elapsed_ms})

        except Exception:
            tb = traceback.format_exc()
            print(f"  ERROR {label}\n{tb}")
            all_pass = False
            records.append({"shape": str(shape), "status": "ERROR",
                             "traceback": tb})

    # Write JSON result
    result = {
        "kernel": kernel_name,
        "overall": "PASS" if all_pass else "FAIL",
        "shapes": records,
    }
    out_path = Path(results_dir) / f"{kernel_name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult: {'PASS ✓' if all_pass else 'FAIL ✗'}  → {out_path}")
    print(f"{'='*60}\n")

    return all_pass


# ── retry wrapper ─────────────────────────────────────────────────────────────
def retry_until_pass(implement_fn, test_fn, max_retries=20):
    """
    implement_fn() : callable — re-imports / reloads the tilelang_impl module
    test_fn()      : callable() -> bool — returns True on pass
    """
    for attempt in range(1, max_retries + 1):
        print(f"\n--- Attempt {attempt}/{max_retries} ---")
        try:
            impl = implement_fn()
            passed = test_fn(impl)
            if passed:
                print(f"✓ Passed on attempt {attempt}")
                return True
        except Exception:
            traceback.print_exc()
    print(f"✗ Failed after {max_retries} attempts")
    return False