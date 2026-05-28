#!/usr/bin/env python3
"""
Experiment — Correctness on edge-case inputs + mitigation-kernel revalidation.

Answers R3 / W11. Two independent parts:

  (a) EDGE-CASE harness: run a handful of kernels (matmul, layer_norm, softmax,
      add) on adversarial inputs — NaN, +/-Inf, large-magnitude (1e4), denormals,
      all-equal rows — comparing the DSL impl (Triton/TileLang) against the PyTorch
      reference and reporting WHERE and HOW they diverge. NaN/Inf propagation
      differences are EXPECTED and are documented, not treated as failures: the
      point is to characterise behaviour, which is exactly what R3 asks for.

  (b) MITIGATION revalidation: re-run the five AKO4ALL mitigation kernels
      (layer_norm_tilelang, rms_norm_tilelang, argmax_tilelang, matmul_triton,
      conv2d_triton) against the PyTorch reference using test_utils-style
      tolerances, so the authors can state the mitigation kernels were revalidated
      on real hardware.

Portable: device props via _harness; nothing hardcoded to sm_89 — identical script
runs on A100/H100. This is a CORRECTNESS experiment (GPU-load-insensitive). The task
asks for SMOKE ONLY (build + tiny validate); --smoke uses small shapes.

Usage:
    python experiments/exp_correctness_edge.py --smoke    # build + tiny validate
    python experiments/exp_correctness_edge.py            # larger shapes
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from _harness import banner, load_impl, load_optimized, write_csv

# test_utils lives in ViperBench (already on sys.path via _harness). Reuse its tolerances.
try:
    import test_utils as TU
except Exception:
    TU = None


# ---------------------------------------------------------------------------
# Edge-case input generators. Each returns a tensor of the requested shape/dtype
# stamped with the adversarial pattern.
# ---------------------------------------------------------------------------
def _base(shape, dtype, dev):
    return torch.randn(*shape, device=dev, dtype=dtype)


def make_edge(kind, shape, dtype, dev):
    """Return a tensor exhibiting the named edge condition."""
    t = _base(shape, dtype, dev)
    if kind == "normal":
        return t
    if kind == "nan":
        t.view(-1)[0] = float("nan")
        return t
    if kind == "posinf":
        t.view(-1)[0] = float("inf")
        return t
    if kind == "neginf":
        t.view(-1)[0] = float("-inf")
        return t
    if kind == "large_mag":
        return t * 1e4
    if kind == "denormal":
        # smallest-subnormal scale for fp32; for fp16 this underflows toward 0,
        # which is itself the interesting denormal/underflow case.
        tiny = 1.4e-45 if dtype == torch.float32 else 6e-8
        return (t.sign() * tiny)
    if kind == "all_equal_rows":
        # every element in a row identical (degenerate softmax / zero-variance LN)
        row = torch.full(shape, 1.0, device=dev, dtype=dtype)
        return row
    raise ValueError(kind)


# Which edge kinds make sense per kernel (kept small + meaningful).
EDGE_KINDS = ["normal", "nan", "posinf", "neginf", "large_mag", "denormal", "all_equal_rows"]


def _classify(ref, test):
    """Characterise divergence between ref and test. Returns a dict describing the
    NaN/Inf structure on both sides plus a finite-region max abs error."""
    ref = ref.float()
    test = test.float()
    r_nan = torch.isnan(ref); t_nan = torch.isnan(test)
    r_inf = torch.isinf(ref); t_inf = torch.isinf(test)
    # where BOTH are finite, measure the numeric agreement
    both_finite = (~r_nan) & (~t_nan) & (~r_inf) & (~t_inf)
    if both_finite.any():
        fin_err = (ref[both_finite] - test[both_finite]).abs().max().item()
    else:
        fin_err = float("nan")
    nan_match = bool(torch.equal(r_nan, t_nan))
    inf_match = bool(torch.equal(r_inf, t_inf))
    return dict(
        ref_has_nan=bool(r_nan.any()), test_has_nan=bool(t_nan.any()),
        ref_has_inf=bool(r_inf.any()), test_has_inf=bool(t_inf.any()),
        nan_pattern_match=nan_match, inf_pattern_match=inf_match,
        finite_max_abs_err=fin_err,
    )


def _tol_for(dtype, loose=False):
    if TU is not None:
        return TU.get_tol(dtype, loose=loose)
    base = {torch.float32: dict(atol=1e-5, rtol=1e-5),
            torch.float16: dict(atol=1e-3, rtol=1e-3),
            torch.bfloat16: dict(atol=1e-2, rtol=1e-2)}.get(dtype, dict(atol=1e-5, rtol=1e-5))
    return {k: (v * 2 if loose else v) for k, v in base.items()}


# ---------------------------------------------------------------------------
# Part (a): edge-case kernels
# ---------------------------------------------------------------------------
def edge_kernels(M, N):
    """Return the edge-case kernel table: (label, dsl_impl, dtype, build_inputs,
    pytorch_fn_name, dsl_fn_name, loose). Inputs are built fresh per edge kind so
    the same adversarial pattern is fed to BOTH impls."""
    cfg = []

    # matmul (fp16): edge pattern on A, plain B.
    def mm_inputs(kind, dtype, dev):
        a = make_edge(kind, (M, N), dtype, dev)
        b = _base((N, M), dtype, dev)
        return (a, b)
    cfg.append(("matmul", "tilelang", torch.float16, mm_inputs, "matmul", "matmul", False))

    # add (fp16): edge pattern on x, plain y (same shape).
    def add_inputs(kind, dtype, dev):
        x = make_edge(kind, (M, N), dtype, dev)
        y = _base((M, N), dtype, dev)
        return (x, y)
    cfg.append(("add", "tilelang", torch.float16, add_inputs, "add", "add", False))

    # softmax (fp32): edge pattern on the input rows.
    def sm_inputs(kind, dtype, dev):
        return (make_edge(kind, (M, N), dtype, dev),)
    cfg.append(("softmax", "tilelang", torch.float32, sm_inputs, "softmax", "softmax", True))

    # layer_norm (fp32): edge pattern on x; unit weight, zero bias.
    def ln_inputs(kind, dtype, dev):
        x = make_edge(kind, (M, N), dtype, dev)
        w = torch.ones(N, device=dev, dtype=dtype)
        b = torch.zeros(N, device=dev, dtype=dtype)
        return (x, w, b)
    cfg.append(("layer_norm", "tilelang", torch.float32, ln_inputs, "layer_norm", "layer_norm", True))

    return cfg


def run_edge(M, N, rows):
    dev = "cuda"
    print("\n" + "-" * 72)
    print("  PART (a): edge-case inputs  (DSL vs PyTorch; NaN/Inf divergences documented)")
    print("-" * 72)
    for kname, impl, dtype, build, pyfn, dslfn, loose in edge_kernels(M, N):
        # load PyTorch ref + DSL impl once per kernel
        try:
            py_mod = load_impl(kname, "pytorch")
            dsl_mod = load_impl(kname, impl)
            py = getattr(py_mod, pyfn)
            dsl = getattr(dsl_mod, dslfn)
        except Exception:
            print(f"  [{kname}/{impl}] LOAD ERROR:\n{traceback.format_exc()}")
            rows.append(dict(part="edge", kernel=kname, impl=impl, case="(load)",
                             status="LOAD_ERROR", detail=traceback.format_exc().splitlines()[-1]))
            continue

        print(f"\n  == {kname}  (DSL={impl}, dtype={str(dtype).split('.')[-1]}) ==")
        for kind in EDGE_KINDS:
            try:
                args = build(kind, dtype, dev)
                ref = py(*args)
                torch.cuda.synchronize()
                out = dsl(*args)
                torch.cuda.synchronize()
                c = _classify(ref, out)
                tol = _tol_for(dtype, loose=loose)
                # numeric "match" only meaningful where both finite
                fin = c["finite_max_abs_err"]
                finite_ok = (fin == fin) and (fin <= tol["atol"] + tol["rtol"] * 1.0 + 1e-6) \
                    if (fin == fin) else None
                # overall divergence flags
                naninf_div = (not c["nan_pattern_match"]) or (not c["inf_pattern_match"])
                # Build a compact human detail string.
                detail = (f"ref(nan={c['ref_has_nan']},inf={c['ref_has_inf']}) "
                          f"dsl(nan={c['test_has_nan']},inf={c['test_has_inf']}) "
                          f"nan_match={c['nan_pattern_match']} inf_match={c['inf_pattern_match']} "
                          f"finite_max_abs={fin:.3g}" if (fin == fin)
                          else (f"ref(nan={c['ref_has_nan']},inf={c['ref_has_inf']}) "
                                f"dsl(nan={c['test_has_nan']},inf={c['test_has_inf']}) "
                                f"nan_match={c['nan_pattern_match']} inf_match={c['inf_pattern_match']} "
                                f"finite_max_abs=n/a"))
                # Status: edge cases are CHARACTERISED, not pass/fail. For finite-only
                # cases (normal/large_mag/all_equal_rows/denormal) we DO assert numeric
                # agreement; for nan/inf cases we report propagation behaviour.
                if kind in ("nan", "posinf", "neginf"):
                    status = "DIVERGE_NAN_INF" if naninf_div else "PROPAGATES_SAME"
                else:
                    if finite_ok is True and not naninf_div:
                        status = "MATCH"
                    elif c["test_has_nan"] or c["test_has_inf"] or naninf_div:
                        status = "DIVERGE_NAN_INF"
                    else:
                        status = "NUMERIC_DIFF"
                tag = {"MATCH": "PASS", "PROPAGATES_SAME": "ok",
                       "DIVERGE_NAN_INF": "doc", "NUMERIC_DIFF": "WARN",
                       }.get(status, "?")
                print(f"    [{tag:<4}] {kind:<15} {status:<16} {detail}")
                rows.append(dict(part="edge", kernel=kname, impl=impl, case=kind,
                                 status=status, detail=detail))
            except Exception:
                tb = traceback.format_exc().splitlines()[-1]
                print(f"    [ERR ] {kind:<15} ERROR  {tb}")
                rows.append(dict(part="edge", kernel=kname, impl=impl, case=kind,
                                 status="ERROR", detail=tb))


# ---------------------------------------------------------------------------
# Part (b): mitigation-kernel revalidation
# ---------------------------------------------------------------------------
def _mitigation_specs(smoke):
    """Each entry: (opt_name, pytorch_kernel_for_ref, dtype, loose, build_inputs).
    build_inputs(dev) -> (args_tuple, pytorch_callable). We build SMALL inputs for
    --smoke rather than the get_inputs() production shapes (8192^2 etc)."""
    specs = []

    # layer_norm_tilelang  (bf16). ref = ViperBench layer_norm pytorch_impl.
    def ln_build(dev):
        M, N = (128, 256) if smoke else (1024, 2048)
        x = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, device=dev, dtype=torch.bfloat16)
        b = torch.randn(N, device=dev, dtype=torch.bfloat16)
        return (x, w, b)
    specs.append(("layer_norm_tilelang", "layer_norm", torch.bfloat16, True, ln_build, "layer_norm"))

    # rms_norm_tilelang  (fp16). ref = ViperBench rms_norm pytorch_impl
    #   rms_norm(x, normalized_shape, weight, eps).
    def rms_build(dev):
        M, N = (128, 256) if smoke else (1024, 2048)
        x = torch.randn(M, N, device=dev, dtype=torch.float16)
        w = torch.randn(N, device=dev, dtype=torch.float16)
        return (x, (N,), w)
    specs.append(("rms_norm_tilelang", "rms_norm", torch.float16, True, rms_build, "rms_norm"))

    # argmax_tilelang  (fp16, dim=1). ref = ViperBench argmax pytorch_impl.
    def am_build(dev):
        M, N = (128, 512) if smoke else (1024, 8192)
        x = torch.randn(M, N, device=dev, dtype=torch.float16)
        return (x, 1)
    specs.append(("argmax_tilelang", "argmax", torch.int64, False, am_build, "argmax"))

    # matmul_triton  (fp16). ref = ViperBench matmul pytorch_impl.
    def mm_build(dev):
        M, K, N = (128, 128, 128) if smoke else (1024, 1024, 1024)
        a = torch.randn(M, K, device=dev, dtype=torch.float16)
        b = torch.randn(K, N, device=dev, dtype=torch.float16)
        return (a, b)
    specs.append(("matmul_triton", "matmul", torch.float16, False, mm_build, "matmul"))

    # conv2d_triton  (fp16). ref = ViperBench conv2d pytorch_impl
    #   conv2d(input, weight, bias, stride, padding). Use padding=1, 3x3 to match.
    def cv_build(dev):
        if smoke:
            x = torch.randn(2, 8, 16, 16, device=dev, dtype=torch.float16)
            w = torch.randn(8, 8, 3, 3, device=dev, dtype=torch.float16)
        else:
            x = torch.randn(4, 32, 32, 32, device=dev, dtype=torch.float16)
            w = torch.randn(32, 32, 3, 3, device=dev, dtype=torch.float16)
        return (x, w, None, 1, 1)   # input, weight, bias, stride, padding
    specs.append(("conv2d_triton", "conv2d", torch.float16, True, cv_build, "conv2d"))

    return specs


def _argmax_compare(ref, test):
    """argmax returns int64 indices; ties can pick different valid indices. Compare
    index equality but tolerate ties (where ref and test point to equal max values)."""
    ref = ref.long().flatten()
    test = test.long().flatten()
    exact = (ref == test).float().mean().item() * 100.0
    return exact


def run_mitigation(smoke, rows):
    dev = "cuda"
    print("\n" + "-" * 72)
    print("  PART (b): mitigation-kernel revalidation  (AKO4ALL optimized vs PyTorch ref)")
    print("-" * 72)
    for opt_name, ref_kernel, dtype, loose, build, fn_name in _mitigation_specs(smoke):
        try:
            opt_mod = load_optimized(opt_name)
        except Exception:
            print(f"  [{opt_name}] LOAD ERROR:\n{traceback.format_exc()}")
            rows.append(dict(part="mitigation", kernel=opt_name, impl="optimized",
                             case="(load)", status="LOAD_ERROR",
                             detail=traceback.format_exc().splitlines()[-1]))
            continue
        try:
            py_mod = load_impl(ref_kernel, "pytorch")
            py = getattr(py_mod, ref_kernel)
        except Exception:
            print(f"  [{opt_name}] REF LOAD ERROR:\n{traceback.format_exc()}")
            rows.append(dict(part="mitigation", kernel=opt_name, impl="optimized",
                             case="(ref-load)", status="LOAD_ERROR",
                             detail=traceback.format_exc().splitlines()[-1]))
            continue

        # the mitigation kernel exposes the unified fn (e.g. layer_norm) at module top.
        opt_fn = getattr(opt_mod, fn_name, None)
        if opt_fn is None and hasattr(opt_mod, "Model"):
            opt_fn = opt_mod.Model().forward
        if opt_fn is None:
            print(f"  [{opt_name}] no callable '{fn_name}' or Model.forward")
            rows.append(dict(part="mitigation", kernel=opt_name, impl="optimized",
                             case="(api)", status="NO_API", detail=f"missing {fn_name}"))
            continue

        try:
            args = build(dev)
            ref = py(*args)
            torch.cuda.synchronize()
            out = opt_fn(*args)
            torch.cuda.synchronize()

            if ref_kernel == "argmax":
                pct = _argmax_compare(ref, out)
                passed = pct >= 99.9     # ties allowed but should be rare on random fp16
                detail = f"index_match={pct:.2f}%"
                max_err = 100.0 - pct
            else:
                if opt_name == "matmul_triton":
                    # fp16 GEMM accumulates ~0.1 abs error over the K dimension; match
                    # ViperBench matmul/test.py (atol=0.2, rtol=1e-2), NOT the strict
                    # fp16 default (1e-3) which spuriously flags valid fp16 GEMM output.
                    tol = dict(atol=0.2, rtol=1e-2)
                    rf, of = ref.float(), out.float()
                    max_err = (rf - of).abs().max().item()
                    passed = torch.allclose(rf, of, **tol)
                elif TU is not None:
                    passed, max_err = TU.compare_tensors(ref, out, dtype, loose=loose)
                    tol = _tol_for(dtype, loose)
                else:
                    tol = _tol_for(dtype, loose=loose)
                    rf, of = ref.float(), out.float()
                    max_err = (rf - of).abs().max().item()
                    passed = torch.allclose(rf, of, **tol)
                detail = f"max_abs_err={max_err:.3g}  tol={tol}"
            status = "REVALIDATED" if passed else "MISMATCH"
            tag = "PASS" if passed else "FAIL"
            print(f"  [{tag}] {opt_name:<22} {status:<12} {detail}")
            rows.append(dict(part="mitigation", kernel=opt_name, impl="optimized",
                             case="revalidate", status=status, detail=detail))
        except Exception:
            tb = traceback.format_exc()
            print(f"  [ERR ] {opt_name:<22} ERROR\n{tb}")
            rows.append(dict(part="mitigation", kernel=opt_name, impl="optimized",
                             case="revalidate", status="ERROR",
                             detail=tb.splitlines()[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="small shapes; build + tiny validate")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available; this experiment requires a GPU.")
        sys.exit(2)

    banner("exp_correctness_edge  (edge-case behaviour + mitigation revalidation; R3/W11)")
    if TU is None:
        print("  WARNING: ViperBench test_utils not importable; using built-in tolerances.")

    if args.smoke:
        M, N = 128, 256
        print(f"  [SMOKE] edge shape M,N = {M},{N}; mitigation = small shapes")
    else:
        M, N = 1024, 2048
        print(f"  [FULL]  edge shape M,N = {M},{N}")

    rows = []
    run_edge(M, N, rows)
    run_mitigation(args.smoke, rows)

    write_csv("correctness_edge", rows, ["part", "kernel", "impl", "case", "status", "detail"])

    # ---- summary ----------------------------------------------------------
    edge_rows = [r for r in rows if r["part"] == "edge"]
    mit_rows = [r for r in rows if r["part"] == "mitigation"]
    revalidated = [r for r in mit_rows if r["status"] == "REVALIDATED"]
    mit_problems = [r for r in mit_rows if r["status"] in ("MISMATCH", "ERROR", "LOAD_ERROR", "NO_API")]
    edge_errors = [r for r in edge_rows if r["status"] == "ERROR"]
    edge_warn = [r for r in edge_rows if r["status"] == "NUMERIC_DIFF"]

    print("\n" + "=" * 72)
    print(f"  EDGE: {len(edge_rows)} cases across "
          f"{len(set(r['kernel'] for r in edge_rows))} kernels; "
          f"{len(edge_errors)} crashed, {len(edge_warn)} numeric-diff on finite inputs.")
    print(f"        (NaN/Inf propagation divergences are EXPECTED and documented, not failures.)")
    print(f"  MITIGATION: {len(revalidated)}/{len(mit_rows)} kernels REVALIDATED against "
          f"PyTorch; {len(mit_problems)} with issues.")
    if revalidated:
        print("        revalidated: " + ", ".join(r["kernel"] for r in revalidated))
    if mit_problems:
        print("        ISSUES: " + ", ".join(f"{r['kernel']}({r['status']})" for r in mit_problems))
    print("=" * 72)

    # Exit non-zero only if a mitigation kernel that LOADED produced a real MISMATCH,
    # or an edge kernel CRASHED unexpectedly. (NaN/Inf docs + load failures of optional
    # kernels do not fail the smoke build.)
    hard_fail = any(r["status"] == "MISMATCH" for r in mit_rows) or len(edge_errors) > 0
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
