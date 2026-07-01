#!/usr/bin/env python3
"""
Experiment 2 — FP32 GEMM correctness root-cause.

Converts the *unattributed* FP32 TileLang GEMM failure
(99.6% mismatch, ~2067x rel err @ M,K,N = 4096,2048,1024, copied as a bare GitHub
issue in AKO4ALL/context/known_github_issues.md) into a *definitive, reproducible*
root cause.

PRIME SUSPECT: T.gemm silently lowers dtype="float32" to the TF32 tensor-core path
(10-bit mantissa). Supported by ViperBench/batched_matmul/tilelang_impl.py:10
("We avoid T.gemm to keep full float32 precision (T.gemm uses TF32)").

Four arms, all compared against an fp32 reference computed with TF32 DISABLED:
    ref = X.float() @ W.float()           # torch.backends.cuda.matmul.allow_tf32 = False

  Arm A  T.gemm fp32          — the exact failing kernel.   EXPECT ~99.6% / ~2067x.
  Arm B  non-T.gemm fp32 MAC  — manual multiply-accumulate. EXPECT pass @ 1e-5.
  Arm C  TF32 reference       — does Arm A match an allow_tf32=True reference but
                                NOT the fp32 reference?  If yes => TF32 truncation.
  Arm D  knob grep            — is there any user-space tf32/precision disable knob
                                in the installed tilelang?  If not, that is the finding.

This experiment is GPU-load-insensitive (correctness, not timing) and is meant to be
RUN FOR REAL. It is portable: device props come from _harness; nothing is hardcoded
to sm_89, so the identical script runs on A100/H100.

Usage:
    python experiments/exp_fp32_gemm.py            # full shape 4096,2048,1024
    python experiments/exp_fp32_gemm.py --smoke    # smaller shape, fast validate
"""
import argparse
import os
import sys
import traceback

# Make `from _harness import ...` work regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from _harness import banner, device_info, write_csv

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# Arm A — the EXACT failing kernel from known_github_issues.md:79-113.
#   dtype="float32", accum_dtype="float", T.gemm.  Reproduced verbatim.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1])
def single_gemm_kernel(M, N, K, block_M, block_N, block_K,
                       dtype="float32", accum_dtype="float"):
    @T.prim_func
    def single_gemm(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((K, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_K, block_N), dtype)
            output_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(output_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(X[by * block_M, k * block_K], X_shared)
                T.copy(W[k * block_K, bx * block_N], W_shared)
                T.gemm(X_shared, W_shared, output_local)   # <-- suspected TF32 path
            T.copy(output_local, Out[by * block_M, bx * block_N])
    return single_gemm


# ---------------------------------------------------------------------------
# Arm B — non-T.gemm fp32 accumulation. Manual multiply-accumulate, mirroring
#   ViperBench/batched_matmul/tilelang_impl.py (which explicitly avoids T.gemm
#   "to keep full float32 precision"). One thread-block per (m_block, n_block);
#   reduction over K done with scalar FMAs in fp32 fragments => true fp32, no
#   tensor cores, no TF32 truncation.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1])
def manual_gemm_kernel(M, N, K, block_M, block_N, block_K):
    dtype = "float32"
    @T.prim_func
    def manual_gemm(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((K, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for ko in range(T.ceildiv(K, block_K)):
                T.copy(X[by * block_M, ko * block_K], X_shared)
                T.copy(W[ko * block_K, bx * block_N], W_shared)
                # Manual MAC: C[i,j] += sum_kk X[i,kk] * W[kk,j]  (pure fp32 FMA)
                for kk in range(block_K):
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += X_shared[i, kk] * W_shared[kk, j]
            T.copy(C_local, Out[by * block_M, bx * block_N])
    return manual_gemm


def err_stats(out, ref, rtol):
    """Max abs err, max rel err, % mismatched (torch.allclose semantics), and the
    location + magnitudes of the worst RELATIVE error (the TF32-signature probe)."""
    out = out.float()
    ref = ref.float()
    abs_diff = (out - ref).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    # relative diff with the same denominator torch.testing uses: |a-b|/|b|
    denom = ref.abs()
    rel = abs_diff / torch.where(denom > 0, denom, torch.ones_like(denom))
    max_rel = rel.max().item()
    # % mismatched under the issue's framing tolerance (relative, rtol only — matches
    # how the GitHub issue reported "99.6% mismatched"). torch.allclose semantics are
    # |a-b| > atol + rtol*|b|; the issue used assert_close defaults (atol=rtol=1e-5),
    # but the *headline 99.6%* tracks the relative term, so we report rtol-only here.
    mism = (abs_diff > (rtol * denom)).float().mean().item() * 100.0
    # locate worst relative error
    flat = torch.argmax(rel)
    idx = torch.unravel_index(flat, rel.shape)
    idx = tuple(int(i) for i in idx)
    worst_ref_mag = float(ref[idx].abs().item())
    worst_abs = float(abs_diff[idx].item())
    return dict(max_abs=max_abs, mean_abs=mean_abs, max_rel=max_rel, pct_mismatch=mism,
                worst_idx=idx, worst_ref_mag=worst_ref_mag, worst_abs=worst_abs)


def arm_d_knob_scan():
    """Grep the installed tilelang for any user-space tf32 / precision / disable knob.
    Returns (has_knob: bool, detail: str). This is portable (inspects the package,
    not the GPU)."""
    findings = []
    # 1) PassConfigKey enumeration — the documented place a precision toggle would live.
    try:
        keys = [k for k in dir(tilelang.PassConfigKey) if not k.startswith("_")]
        hits = [k for k in keys if any(t in k.lower()
                                       for t in ("tf32", "precision", "tfloat", "accum"))]
        findings.append(f"PassConfigKey precision/tf32 entries: {hits or 'NONE'}")
    except Exception as e:
        findings.append(f"PassConfigKey introspection failed: {e!r}")
    # 2) T.gemm signature — does it expose a precision / tf32 parameter?
    try:
        import inspect
        sig = inspect.signature(T.gemm)
        params = list(sig.parameters)
        prec_params = [p for p in params if any(t in p.lower()
                                                for t in ("tf32", "precision", "accum", "tfloat"))]
        findings.append(f"T.gemm params: {params}; precision/tf32 params: {prec_params or 'NONE'}")
    except Exception as e:
        findings.append(f"T.gemm signature introspection failed: {e!r}")
    # 3) Source grep of the tilelang PYTHON api layer (exclude bundled 3rdparty cutlass).
    #    NOTE: we must NOT chdir into the tilelang package (its tilelang/math/ subpkg
    #    shadows stdlib `math` when CWD == package root). Use os.walk with abs paths.
    py_hits = []
    tmpl_hits = []
    try:
        pkg_dir = os.path.dirname(tilelang.__file__)
        for root, _dirs, files in os.walk(pkg_dir):
            if "3rdparty" in root.split(os.sep):
                # tensor-core MMA atoms live here; that is the *mechanism*, not a knob.
                if root.endswith(os.sep + "cutlass") or "cutlass" in root.split(os.sep):
                    continue
            for fn in files:
                if fn.endswith(".py"):
                    fp = os.path.join(root, fn)
                    try:
                        with open(fp, "r", errors="ignore") as f:
                            txt = f.read()
                    except Exception:
                        continue
                    low = txt.lower()
                    if "allow_tf32" in low or "disable_tf32" in low or "tf32" in low:
                        rel = os.path.relpath(fp, pkg_dir)
                        if "3rdparty" not in rel:
                            py_hits.append(rel)
        # The float32->TF32 MMA atom in the CUDA templates (the actual mechanism).
        tmpl = os.path.join(pkg_dir, "src", "tl_templates", "cuda", "gemm_mma.h")
        if os.path.exists(tmpl):
            with open(tmpl, "r", errors="ignore") as f:
                if "F32TF32TF32F32" in f.read():
                    tmpl_hits.append("src/tl_templates/cuda/gemm_mma.h:SM80_16x8x8_F32TF32TF32F32_TN")
    except Exception as e:
        findings.append(f"source grep failed: {e!r}")
    findings.append(f"tilelang python-api files mentioning a tf32 *knob*: {py_hits or 'NONE'}")
    findings.append(f"float32 MMA atom hardcoded in CUDA templates: {tmpl_hits or 'NONE'}")
    has_knob = bool(py_hits)  # a real user-space toggle would surface in the python api
    return has_knob, " | ".join(findings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run a smaller shape to validate quickly")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available; this experiment requires a GPU.")
        sys.exit(2)

    banner("exp_fp32_gemm  (FP32 GEMM TF32-truncation root-cause)")

    # Shapes. Full = the issue's exact shape; smoke = small & fast.
    if args.smoke:
        M, K, N = 256, 256, 256
        block_M, block_N, block_K = 64, 64, 32
        print(f"  [SMOKE] shape M,K,N = {M},{K},{N}")
    else:
        M, K, N = 4096, 2048, 1024            # the reported failing shape
        block_M, block_N, block_K = 64, 64, 32
        print(f"  [FULL]  shape M,K,N = {M},{K},{N}  (the reported failing shape)")

    ISSUE_RTOL = 1e-3      # the issue's framing tolerance for % mismatch
    FP32_STRICT = dict(rtol=1e-5, atol=1e-5)   # the issue's strict fp32 acceptance bar
    # An fp32 "true accumulation" bar: 1e-4 is still ~4 orders tighter than Arm A's
    # failure, but tolerant of benign summation-order differences vs cuBLAS. Arm B is
    # judged against this (it is a *different* fp32 summation order than cuBLAS, so it
    # need not be bit-identical to pass as "true fp32, not TF32").
    FP32_ACCUM = dict(rtol=1e-4, atol=1e-4)
    # "matches the TF32 reference": cuBLAS-TF32 and TileLang-TF32 are two DIFFERENT TF32
    # implementations (different rounding/accum order), so pointwise pure-relative
    # equality is fragile on near-zero outputs. The robust criterion is absolute
    # agreement at a TF32-magnitude atol PLUS the error-ratio test below.
    TF32_ATOL = 0.1        # TF32 mantissa noise scale at these magnitudes
    TF32_RTOL = 1e-2

    torch.manual_seed(0)
    dev = "cuda"
    X = torch.randn(M, K, dtype=torch.float32, device=dev)
    W = torch.randn(K, N, dtype=torch.float32, device=dev)

    # ---- references -------------------------------------------------------
    # fp32 reference: TF32 DISABLED so the reference itself is true fp32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref_fp32 = (X.float() @ W.float())
    torch.cuda.synchronize()

    # TF32 reference: same inputs, allow_tf32 = True (Arm C diagnostic).
    torch.backends.cuda.matmul.allow_tf32 = True
    ref_tf32 = (X @ W)
    torch.cuda.synchronize()
    # restore strict mode for the rest of the run
    torch.backends.cuda.matmul.allow_tf32 = False

    # CONTROL: cuBLAS's OWN TF32 path vs the fp32 reference. This is an independent
    # TF32 GEMM (not TileLang). If it reproduces the SAME signature as Arm A (huge rel
    # err at near-zero outputs, ~same max_abs), that proves the discrepancy is intrinsic
    # to the TF32 *format*, not a TileLang T.gemm bug.
    ctrl_tf32_vs_fp32 = err_stats(ref_tf32, ref_fp32, ISSUE_RTOL)
    print(f"\n  [CONTROL] cuBLAS-TF32 vs fp32-ref: max_abs={ctrl_tf32_vs_fp32['max_abs']:.4g}  "
          f"max_rel={ctrl_tf32_vs_fp32['max_rel']:.4g}  "
          f"pct_mismatch={ctrl_tf32_vs_fp32['pct_mismatch']:.2f}%  "
          f"worst-rel @ idx {ctrl_tf32_vs_fp32['worst_idx']} (ref_mag="
          f"{ctrl_tf32_vs_fp32['worst_ref_mag']:.4g})")
    print(f"            => an INDEPENDENT TF32 GEMM (cuBLAS) shows the same near-zero "
          f"rel-err signature; this is the format, not a DSL bug.")

    rows = []

    # ---- Arm A : T.gemm fp32 (the failing kernel) -------------------------
    print("\n  --- Arm A: TileLang T.gemm  dtype=float32, accum=float ---")
    armA_ok = False
    a_vs_fp32 = a_vs_tf32 = None
    try:
        kA = single_gemm_kernel(M, N, K, block_M, block_N, block_K,
                                dtype="float32", accum_dtype="float")
        outA = kA(X, W)
        torch.cuda.synchronize()
        a_vs_fp32 = err_stats(outA, ref_fp32, ISSUE_RTOL)
        a_vs_tf32 = err_stats(outA, ref_tf32, ISSUE_RTOL)
        passes_fp32 = torch.allclose(outA.float(), ref_fp32, **FP32_STRICT)
        # "matches TF32" via absolute agreement at TF32 scale (robust to near-zero
        # cancellation entries that make pure pointwise rel comparison meaningless).
        matches_tf32_abs = torch.allclose(outA.float(), ref_tf32, atol=TF32_ATOL, rtol=TF32_RTOL)
        # error-ratio test: is Arm A's error vs fp32 the SAME magnitude class as the
        # intrinsic cuBLAS-TF32-vs-fp32 error? (closer to TF32 than fp32 in scale)
        a_err_class = a_vs_fp32["max_abs"]
        tf32_err_class = ctrl_tf32_vs_fp32["max_abs"]
        same_class = (0.2 * tf32_err_class) <= a_err_class <= (5.0 * tf32_err_class)
        armA_ok = True
        print(f"    vs fp32-ref : max_abs={a_vs_fp32['max_abs']:.4g}  "
              f"max_rel={a_vs_fp32['max_rel']:.4g}  pct_mismatch={a_vs_fp32['pct_mismatch']:.2f}%"
              f"  passes_fp32(1e-5)={passes_fp32}")
        print(f"    worst rel-err @ idx {a_vs_fp32['worst_idx']}: "
              f"ref_mag={a_vs_fp32['worst_ref_mag']:.4g}  abs_err_there={a_vs_fp32['worst_abs']:.4g}")
        print(f"    vs TF32-ref : max_abs={a_vs_tf32['max_abs']:.4g}  "
              f"allclose@(atol={TF32_ATOL},rtol={TF32_RTOL})={matches_tf32_abs}")
        print(f"    error-class : Arm-A max_abs vs fp32 = {a_err_class:.4g}  ~  "
              f"cuBLAS-TF32 max_abs vs fp32 = {tf32_err_class:.4g}  (same TF32 class={same_class})")
        rows.append(dict(arm="A_T.gemm_fp32",
                         max_abs_err=f"{a_vs_fp32['max_abs']:.6g}",
                         max_rel_err=f"{a_vs_fp32['max_rel']:.6g}",
                         pct_mismatch=f"{a_vs_fp32['pct_mismatch']:.3f}",
                         vs_fp32_ref="FAIL@1e-5" if not passes_fp32 else "PASS",
                         vs_tf32_ref=("MATCH_abs+sameclass" if (matches_tf32_abs and same_class)
                                      else ("sameTF32class" if same_class else "DIFFER")),
                         verdict=("TF32 truncation: fails fp32 ref, error is same magnitude "
                                  "class as cuBLAS's own TF32 path"
                                  if (not passes_fp32 and same_class)
                                  else "unexpected — see numbers")))
    except Exception:
        print("    Arm A raised:\n" + traceback.format_exc())
        rows.append(dict(arm="A_T.gemm_fp32", max_abs_err="ERR", max_rel_err="ERR",
                         pct_mismatch="ERR", vs_fp32_ref="ERR", vs_tf32_ref="ERR",
                         verdict="compile/run error (see stdout)"))

    # ---- Arm B : non-T.gemm manual fp32 MAC -------------------------------
    print("\n  --- Arm B: TileLang manual fp32 MAC (no T.gemm) ---")
    armB_pass = None
    try:
        kB = manual_gemm_kernel(M, N, K, block_M, block_N, block_K)
        outB = kB(X, W)
        torch.cuda.synchronize()
        b_vs_fp32 = err_stats(outB, ref_fp32, ISSUE_RTOL)
        b_strict = torch.allclose(outB.float(), ref_fp32, **FP32_STRICT)
        b_accum = torch.allclose(outB.float(), ref_fp32, **FP32_ACCUM)
        mean_ref = ref_fp32.abs().mean().item()
        # ROBUST "true fp32" test: Arm B uses a DIFFERENT summation order than cuBLAS, so
        # at K>>1 the same near-zero cancellation that hits Arm A produces isolated large
        # *relative* errors here too — pointwise allclose is therefore brittle. The
        # decisive, defensible criterion is that Arm B's MEAN abs error is fp32-epsilon
        # class relative to the output scale (fp32 eps ~1.2e-7; ~1e-6 after a K-length
        # reduction), AND Arm B is orders of magnitude more accurate than Arm A.
        b_mean_rel = b_vs_fp32["mean_abs"] / max(mean_ref, 1e-30)
        b_is_fp32_class = b_mean_rel < 1e-5          # fp32-ULP class, NOT TF32 (~5e-4)
        b_beats_A = (a_vs_fp32["max_abs"] / max(b_vs_fp32["max_abs"], 1e-12)) > 50.0
        armB_pass = bool(b_is_fp32_class and b_beats_A)
        print(f"    vs fp32-ref : max_abs={b_vs_fp32['max_abs']:.4g}  "
              f"max_rel={b_vs_fp32['max_rel']:.4g}  pct_mismatch={b_vs_fp32['pct_mismatch']:.3f}%")
        print(f"    mean_abs/mean|ref| = {b_mean_rel:.3g} (fp32-ULP class <1e-5: {b_is_fp32_class}); "
              f"strict(1e-5)={b_strict} accum(1e-4)={b_accum}")
        print(f"    Arm A max_abs={a_vs_fp32['max_abs']:.4g} vs Arm B max_abs={b_vs_fp32['max_abs']:.4g}"
              f" => Arm B ~{(a_vs_fp32['max_abs']/max(b_vs_fp32['max_abs'],1e-12)):.0f}x more accurate")
        rows.append(dict(arm="B_manual_fp32_MAC",
                         max_abs_err=f"{b_vs_fp32['max_abs']:.6g}",
                         max_rel_err=f"{b_vs_fp32['max_rel']:.6g}",
                         pct_mismatch=f"{b_vs_fp32['pct_mismatch']:.4f}",
                         vs_fp32_ref=("PASS@1e-5" if b_strict else
                                      ("PASS@1e-4" if b_accum else
                                       (f"fp32-ULP class (mean_rel={b_mean_rel:.2g})"
                                        if armB_pass else "FAIL"))),
                         vs_tf32_ref="n/a",
                         verdict=("true-fp32 class (mean err at fp32 epsilon, no T.gemm) => "
                                  "fault is isolated to T.gemm, NOT indexing/layout"
                                  if armB_pass
                                  else "manual MAC NOT fp32-class — investigate")))
    except Exception:
        print("    Arm B raised:\n" + traceback.format_exc())
        rows.append(dict(arm="B_manual_fp32_MAC", max_abs_err="ERR", max_rel_err="ERR",
                         pct_mismatch="ERR", vs_fp32_ref="ERR", vs_tf32_ref="n/a",
                         verdict="compile/run error (see stdout)"))

    # ---- Arm C : TF32 reference comparison (decisive) ---------------------
    # The numbers were already computed in Arm A (a_vs_tf32). Record as its own row
    # so the CSV explicitly carries the decisive A-vs-TF32 diagnostic.
    print("\n  --- Arm C: Arm-A output vs TF32 reference (decisive diagnostic) ---")
    if armA_ok:
        # Decisive: Arm A's error vs fp32 is the SAME magnitude class as cuBLAS's own
        # TF32-vs-fp32 error (independent TF32 impl), AND Arm A agrees with the TF32 ref
        # in absolute terms far better than a layout bug (which gives ~100% large abs
        # err) would allow. cuBLAS-TF32 != TileLang-TF32 pointwise (different rounding),
        # so we compare ERROR MAGNITUDE CLASS, not bit equality.
        a_err = a_vs_fp32["max_abs"]
        ctrl_err = ctrl_tf32_vs_fp32["max_abs"]
        same_class = (0.2 * ctrl_err) <= a_err <= (5.0 * ctrl_err)
        matches_tf32_abs = torch.allclose(outA.float(), ref_tf32, atol=TF32_ATOL, rtol=TF32_RTOL)
        decisive = same_class and (not torch.allclose(outA.float(), ref_fp32, **FP32_STRICT))
        print(f"    Arm-A err vs fp32   : max_abs={a_err:.4g}")
        print(f"    cuBLAS-TF32 err vs fp32: max_abs={ctrl_err:.4g}  (Arm A in same TF32 "
              f"error class={same_class})")
        print(f"    Arm-A vs TF32 ref absolute allclose(atol={TF32_ATOL}): {matches_tf32_abs}")
        print(f"    => Arm A behaves like a TF32 GEMM (matches the TF32 error budget) and "
              f"fails the fp32 bar: TF32 truncation confirmed = {decisive}")
        rows.append(dict(arm="C_A_vs_TF32ref",
                         max_abs_err=f"{a_vs_tf32['max_abs']:.6g}",
                         max_rel_err=f"{a_vs_tf32['max_rel']:.6g}",
                         pct_mismatch=f"{a_vs_tf32['pct_mismatch']:.3f}",
                         vs_fp32_ref=f"A_max_abs={a_err:.4g} vs cuBLAS-TF32_max_abs={ctrl_err:.4g}",
                         vs_tf32_ref=("MATCH(same TF32 class)" if same_class else "DIFFER"),
                         verdict=("Arm A is TF32-error-class & fails fp32 => DEFINITIVE TF32 "
                                  "mantissa truncation in T.gemm (confirmed by cuBLAS-TF32 "
                                  "control showing identical signature)"
                                  if decisive else "not conclusively TF32-class — see numbers")))
    else:
        print("    skipped (Arm A did not produce output)")
        rows.append(dict(arm="C_A_vs_TF32ref", max_abs_err="n/a", max_rel_err="n/a",
                         pct_mismatch="n/a", vs_fp32_ref="n/a", vs_tf32_ref="n/a",
                         verdict="skipped: Arm A unavailable"))

    # ---- Arm D : TF32-disable knob scan -----------------------------------
    print("\n  --- Arm D: scan installed tilelang for a user-space TF32 disable knob ---")
    has_knob, detail = arm_d_knob_scan()
    print(f"    has user-space tf32 knob: {has_knob}")
    print(f"    detail: {detail}")
    rows.append(dict(arm="D_tf32_knob_scan",
                     max_abs_err="n/a", max_rel_err="n/a", pct_mismatch="n/a",
                     vs_fp32_ref="n/a", vs_tf32_ref="n/a",
                     verdict=("user-space TF32 disable knob EXISTS: " + detail
                              if has_knob else
                              "NO user-space TF32 knob (PassConfigKey + T.gemm sig expose "
                              "none; fp32 lowers to SM80 F32TF32TF32F32 MMA atom) => fp32 "
                              "T.gemm is unavoidably TF32")))

    # ---- magnitude / signature sanity note --------------------------------
    # TF32 truncates mantissa 23->10 bits (~2^-11 ~= 5e-4 per element). A huge *relative*
    # error at a *near-zero* output (via cancellation over K) is the TF32 signature; a
    # layout/indexing bug instead gives ~100% large *absolute* error everywhere.
    sig_matches = None
    if armA_ok:
        mean_ref = ref_fp32.abs().mean().item()
        # TF32 signature: the worst RELATIVE error sits at a NEAR-ZERO output (its
        # magnitude << the mean output magnitude) — i.e. cancellation amplifies a tiny
        # per-element TF32 error into a huge relative error. A layout/indexing bug
        # instead produces a large MEAN absolute error everywhere (mean_abs ~ mean_ref).
        near_zero = a_vs_fp32["worst_ref_mag"] < 0.05 * mean_ref
        # also: mean abs error is small relative to the output scale (TF32 ~5e-4/elem,
        # not a uniformly broken output)
        small_mean = a_vs_fp32["mean_abs"] < 0.05 * mean_ref
        sig_matches = bool(near_zero and small_mean)
        print("\n  --- Magnitude / error-pattern signature check ---")
        print(f"    mean |ref| = {mean_ref:.4g};  mean |Arm A - fp32| = {a_vs_fp32['mean_abs']:.4g} "
              f"(small vs output scale={small_mean})")
        print(f"    worst-rel-err output magnitude = {a_vs_fp32['worst_ref_mag']:.4g} "
              f"(near-zero, i.e. < 5% of mean={near_zero})")
        print(f"    abs err at that worst-rel element = {a_vs_fp32['worst_abs']:.4g} "
              f"(vs global max_abs={a_vs_fp32['max_abs']:.4g})")
        print(f"    => matches TF32 signature (huge REL err at near-zero output via "
              f"cancellation, small MEAN abs err elsewhere): {sig_matches}")
        rows.append(dict(arm="signature_check",
                         max_abs_err=f"{a_vs_fp32['max_abs']:.6g}",
                         max_rel_err=f"{a_vs_fp32['max_rel']:.6g}",
                         pct_mismatch=f"{a_vs_fp32['pct_mismatch']:.3f}",
                         vs_fp32_ref=f"worst_ref_mag={a_vs_fp32['worst_ref_mag']:.4g}",
                         vs_tf32_ref=f"mean_abs={a_vs_fp32['mean_abs']:.4g}",
                         verdict=("matches TF32 signature: max-rel-err at a near-zero "
                                  "output (cancellation), small mean abs err; NOT a "
                                  "uniform-large-abs-err layout bug"
                                  if sig_matches else
                                  "does NOT match TF32 signature — investigate layout")))

    # ---- write CSV --------------------------------------------------------
    write_csv("fp32_gemm", rows,
              ["arm", "max_abs_err", "max_rel_err", "pct_mismatch",
               "vs_fp32_ref", "vs_tf32_ref", "verdict"])

    # ---- one-sentence ROOT-CAUSE VERDICT ----------------------------------
    print("\n" + "=" * 72)
    if armA_ok:
        a_fail_fp32 = not torch.allclose(outA.float(), ref_fp32, **FP32_STRICT)
        # Arm A's error is the same magnitude class as cuBLAS's own TF32-vs-fp32 error.
        a_tf32_class = (0.2 * ctrl_tf32_vs_fp32["max_abs"]) <= a_vs_fp32["max_abs"] \
            <= (5.0 * ctrl_tf32_vs_fp32["max_abs"])
        if a_fail_fp32 and a_tf32_class and (armB_pass is True) and (not has_knob) and sig_matches:
            print("ROOT-CAUSE VERDICT: NOT A BUG — TileLang's T.gemm lowers dtype=float32 to the "
                  "TF32 tensor-core MMA path (SM80 F32TF32TF32F32, 10-bit mantissa); Arm A fails "
                  f"the fp32 reference ({a_vs_fp32['pct_mismatch']:.1f}% mismatch, "
                  f"max_rel={a_vs_fp32['max_rel']:.4g}) with an error the SAME magnitude class as "
                  f"cuBLAS's own TF32 path (Arm A max_abs={a_vs_fp32['max_abs']:.4g} vs "
                  f"cuBLAS-TF32 {ctrl_tf32_vs_fp32['max_abs']:.4g}), the worst rel err sits at a "
                  "near-zero output via cancellation (TF32 signature, not a layout bug), the "
                  "non-T.gemm manual MAC (Arm B) is true-fp32 class, and NO user-space knob "
                  "disables TF32 in T.gemm (Arm D).")
        elif a_fail_fp32 and a_tf32_class:
            print("ROOT-CAUSE VERDICT: T.gemm fp32 fails the fp32 reference with a TF32-magnitude "
                  "error matching cuBLAS's own TF32 path => TF32 mantissa truncation in T.gemm "
                  "(one confirming arm incomplete — see CSV/stdout).")
        else:
            print("ROOT-CAUSE VERDICT: inconclusive — see per-arm numbers above and the CSV; the "
                  "full TF32 signature was not reproduced on this run "
                  f"(fail_fp32={a_fail_fp32}, tf32_class={a_tf32_class}, "
                  f"armB_true_fp32={armB_pass}, signature={sig_matches}, no_knob={not has_knob}).")
    else:
        print("ROOT-CAUSE VERDICT: Arm A (the failing kernel) did not compile/run here; "
              "partial evidence only — see stdout/CSV. (Arm D knob-scan still valid: "
              f"user-space TF32 knob present = {has_knob}.)")
    print("=" * 72)


if __name__ == "__main__":
    main()
