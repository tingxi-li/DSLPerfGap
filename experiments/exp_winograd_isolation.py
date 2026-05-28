#!/usr/bin/env python3
"""
Experiment 4 — Winograd isolation in the conv gap  (answers RC4 / R1)
=====================================================================

cuDNN selects a Winograd algorithm for 3x3 stride-1 convolutions; the Triton and
TileLang conv kernels in this suite only do im2col-GEMM (no Winograd). RC4 claims
part of cuDNN's conv advantage is Winograd, but the artifact has no isolation
experiment. There is no public PyTorch API to force/forbid a specific cuDNN
algorithm by name, so we isolate Winograd's contribution with three complementary,
PyTorch-controllable levers (no raw cuDNN C API):

  (1) Algorithm-selection A/B (upper bound).
      Time cuDNN F.conv2d 3x3 s1 with torch.backends.cudnn.deterministic = False
      vs = True. Deterministic mode forbids most Winograd (and other
      non-deterministic) algorithms, so the (det - nondet) latency delta is an
      UPPER BOUND on Winograd's 3x3 benefit. (benchmark=True so cuDNN actually
      searches; flags are saved/restored.)

  (2) Eligible-vs-ineligible proxy (the cleanest signal).
      Winograd F(2,3)/F(4,3) applies to 3x3 stride-1 but NOT to stride-2 or to
      5x5 / 7x7 filters. Time cuDNN vs Triton vs TileLang at:
          3x3 s1  (Winograd-ELIGIBLE)
          3x3 s2  (ineligible: stride 2)
          5x5 s1  (ineligible: large filter)
          7x7 s1  (ineligible: large filter)
      Report gap = DSL_latency / cuDNN_latency per config. If the DSL-vs-cuDNN gap
      SHRINKS markedly when Winograd is unavailable (s2, 5x5, 7x7) relative to the
      eligible 3x3 s1 case, the residual 3x3 s1 gap is attributable to Winograd.

  (3) Confirm cuDNN's selected algorithm (observability).
      Run ONE conv in a subprocess under CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=stdout
      and grep the cuDNN engine log for "winograd". Turns the assumption into an
      observation (and yields evidence for audit-finding N1: what cuDNN actually
      does). Invoke with:  python exp_winograd_isolation.py --cudnn-log [--shape ...]

Portability
-----------
  * No arch hardcoding; device props from _harness (queried at runtime).
  * Conv shape is parameterized; Ada-safe default input (32,256,128,128) fp16
    (~0.27 GB/tensor, fits ~20 GB). A100/H100 can scale via --batch/--channels/--hw.
  * Per-config OOM is caught and skipped. cuDNN-log step degrades gracefully if the
    debug log is unavailable (older cuDNN / no env support).

Usage
-----
    python exp_winograd_isolation.py            # full sweep (orchestrator)
    python exp_winograd_isolation.py --smoke    # tiny shape, <60s, build-time check
    python exp_winograd_isolation.py --cudnn-log  # confirm cuDNN algo via debug log
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import banner, time_kernel, library_efficiency, load_impl, write_csv  # noqa: E402

EXPERIMENT = "winograd_isolation"

# Ada-safe default conv shape (= the paper's conv-large input).
# input (N,C,H,W) fp16; weight (C_out, C_in, k, k).
DEFAULT = dict(N=32, C=256, H=128, W=128, C_out=256)
SMOKE = dict(N=2, C=32, H=32, W=32, C_out=32)

# (label, kernel_size, stride, winograd_eligible)
CONFIGS = [
    ("3x3_s1", 3, 1, True),    # Winograd-eligible
    ("3x3_s2", 3, 2, False),   # ineligible: stride 2
    ("5x5_s1", 5, 1, False),   # ineligible: large filter
    ("7x7_s1", 7, 1, False),   # ineligible: large filter
]


def make_conv_inputs(dims, k, dtype=torch.float16):
    x = torch.randn(dims["N"], dims["C"], dims["H"], dims["W"],
                    device="cuda", dtype=dtype)
    w = torch.randn(dims["C_out"], dims["C"], k, k, device="cuda", dtype=dtype)
    return x, w


# ---------------------------------------------------------------------------
# (1) cuDNN deterministic A/B  -- upper-bounds Winograd's 3x3 s1 benefit.
# ---------------------------------------------------------------------------
def cudnn_determinism_ab(dims, warmup, reps, rows):
    print("\n--- (1) cuDNN deterministic A/B on 3x3 s1 (delta upper-bounds Winograd) ---")
    cudnn = torch.backends.cudnn
    saved = (cudnn.benchmark, cudnn.deterministic, cudnn.enabled)
    x, w = make_conv_inputs(dims, 3)
    pad = 1  # preserve spatial dims for 3x3 s1
    results = {}
    try:
        cudnn.enabled = True
        cudnn.benchmark = True          # let cuDNN search algorithms (may pick Winograd)
        for det in (False, True):
            cudnn.deterministic = det
            # benchmark cache is keyed on (shape, flags); re-warm after toggling.
            fn = lambda: torch.nn.functional.conv2d(x, w, None, 1, pad, (1, 1), 1)
            try:
                fn(); torch.cuda.synchronize()
                t = time_kernel(fn, warmup=warmup, reps=reps)
                results[det] = t["median_ms"]
                tag = "deterministic(Winograd mostly OFF)" if det else "nondeterministic(Winograd allowed)"
                print(f"      {tag:<40} median={t['median_ms']:.4f} ms  "
                      f"mean={t['mean_ms']:.4f}+/-{t['std_ms']:.4f}")
                rows.append(dict(
                    measurement="cudnn_determinism_ab", config="3x3_s1",
                    impl=f"cudnn_deterministic={det}", winograd_eligible=True,
                    median_ms=t["median_ms"], mean_ms=t["mean_ms"], std_ms=t["std_ms"],
                    gap_vs_cudnn="", e_vs_cudnn="", note=tag))
            except torch.cuda.OutOfMemoryError:
                print(f"      deterministic={det}: OOM"); torch.cuda.empty_cache()
            except Exception as e:
                print(f"      deterministic={det}: ERROR {type(e).__name__}: {str(e)[:90]}")
    finally:
        cudnn.benchmark, cudnn.deterministic, cudnn.enabled = saved

    if False in results and True in results:
        delta = results[True] - results[False]
        ratio = results[True] / results[False] if results[False] > 0 else float("nan")
        print(f"      >> delta (det - nondet) = {delta:+.4f} ms  ({ratio:.3f}x); "
              f"positive => non-deterministic (Winograd-allowed) path is faster, "
              f"upper-bounding Winograd's 3x3 s1 contribution.")
        rows.append(dict(
            measurement="cudnn_determinism_ab", config="3x3_s1",
            impl="DELTA_det_minus_nondet", winograd_eligible=True,
            median_ms=round(delta, 5), mean_ms="", std_ms="",
            gap_vs_cudnn="", e_vs_cudnn=round(ratio, 4),
            note="upper bound on Winograd benefit (det - nondet)"))
    _free(x, w)


# ---------------------------------------------------------------------------
# (2) Eligible-vs-ineligible proxy across cuDNN / Triton / TileLang.
# ---------------------------------------------------------------------------
def eligible_vs_ineligible(dims, warmup, reps, rows):
    print("\n--- (2) Winograd eligible-vs-ineligible proxy (cuDNN vs Triton vs TileLang) ---")
    cudnn_fn = getattr(load_impl("conv2d", "pytorch"), "conv2d")   # F.conv2d -> cuDNN
    triton_fn = getattr(load_impl("conv2d", "triton"), "conv2d")
    try:
        tilelang_fn = getattr(load_impl("conv2d", "tilelang"), "conv2d")
    except Exception as e:
        print(f"  [warn] no tilelang conv2d ({str(e)[:60]}); skipping that arm")
        tilelang_fn = None

    # Ensure cuDNN is free to choose Winograd for the eligible case.
    cudnn = torch.backends.cudnn
    saved = (cudnn.benchmark, cudnn.deterministic)
    cudnn.benchmark, cudnn.deterministic = True, False
    try:
        for label, k, stride, eligible in CONFIGS:
            pad = k // 2  # 'same'-ish padding (preserves spatial dims at stride 1)
            print(f"  config {label} (k={k}, stride={stride}, "
                  f"winograd_eligible={eligible})")
            try:
                x, w = make_conv_inputs(dims, k)
            except torch.cuda.OutOfMemoryError:
                print("      OOM allocating inputs; skip"); torch.cuda.empty_cache(); continue

            impls = [
                ("cudnn", lambda: cudnn_fn(x, w, None, stride, pad, 1)),
                ("triton", lambda: triton_fn(x, w, None, stride, pad, 1)),
            ]
            if tilelang_fn is not None:
                impls.append(("tilelang", lambda: tilelang_fn(x, w, None, stride, pad, 1)))

            timings = {}
            for name, fn in impls:
                try:
                    fn(); torch.cuda.synchronize()
                    t = time_kernel(fn, warmup=warmup, reps=reps)
                    timings[name] = t
                    print(f"      {name:<10} median={t['median_ms']:.4f} ms  "
                          f"mean={t['mean_ms']:.4f}+/-{t['std_ms']:.4f}")
                except torch.cuda.OutOfMemoryError:
                    print(f"      {name:<10} OOM"); torch.cuda.empty_cache(); timings[name] = "oom"
                except Exception as e:
                    print(f"      {name:<10} ERROR {type(e).__name__}: {str(e)[:90]}")
                    timings[name] = "error"

            t_cudnn = timings.get("cudnn")
            for name, _ in impls:
                t = timings[name]
                if isinstance(t, dict):
                    # gap = DSL / cuDNN  (>1 means DSL slower); e_vs_cudnn = cuDNN/DSL*100.
                    gap = (round(t["median_ms"] / t_cudnn["median_ms"], 3)
                           if isinstance(t_cudnn, dict) else "")
                    e = (library_efficiency(t_cudnn["median_ms"], t["median_ms"])
                         if isinstance(t_cudnn, dict) else "")
                    rows.append(dict(
                        measurement="eligible_vs_ineligible", config=label, impl=name,
                        winograd_eligible=eligible,
                        median_ms=t["median_ms"], mean_ms=t["mean_ms"], std_ms=t["std_ms"],
                        gap_vs_cudnn=gap, e_vs_cudnn=e, note=""))
                else:
                    rows.append(dict(
                        measurement="eligible_vs_ineligible", config=label, impl=name,
                        winograd_eligible=eligible,
                        median_ms=t, mean_ms=t, std_ms=t,
                        gap_vs_cudnn=t, e_vs_cudnn=t, note=""))

            # Per-config gap summary.
            for dsl in ("triton", "tilelang"):
                if isinstance(timings.get(dsl), dict) and isinstance(t_cudnn, dict):
                    g = timings[dsl]["median_ms"] / t_cudnn["median_ms"]
                    print(f"      >> {dsl} gap vs cuDNN @ {label}: {g:.2f}x"
                          + ("  [Winograd-eligible]" if eligible else "  [ineligible]"))
            _free(x, w)
    finally:
        cudnn.benchmark, cudnn.deterministic = saved

    # Interpretation hint comparing eligible vs ineligible gaps.
    _print_winograd_interpretation(rows)


def _print_winograd_interpretation(rows):
    def gaps_for(impl):
        out = {}
        for r in rows:
            if r["measurement"] == "eligible_vs_ineligible" and r["impl"] == impl \
                    and isinstance(r["gap_vs_cudnn"], (int, float)):
                out[r["config"]] = r["gap_vs_cudnn"]
        return out
    for impl in ("triton", "tilelang"):
        g = gaps_for(impl)
        if "3x3_s1" in g:
            inel = [g[c] for c in ("3x3_s2", "5x5_s1", "7x7_s1") if c in g]
            if inel:
                avg_inel = sum(inel) / len(inel)
                print(f"  >> {impl}: gap@3x3_s1(eligible)={g['3x3_s1']:.2f}x vs "
                      f"avg gap@ineligible={avg_inel:.2f}x. "
                      + ("Eligible gap LARGER => residual attributable to Winograd."
                         if g["3x3_s1"] > avg_inel else
                         "Eligible gap not larger => Winograd not the dominant factor here."))


# ---------------------------------------------------------------------------
# (3) cuDNN debug-log confirmation (subprocess; env must precede CUDA init).
# ---------------------------------------------------------------------------
def confirm_cudnn_algo(dims, k=3, stride=1):
    """Run ONE conv in a child process under CUDNN_LOGINFO_DBG and grep for
    'winograd'. Returns (found_bool, matching_lines). Documented invocation:
        python exp_winograd_isolation.py --cudnn-log --shape N C H W Cout
    """
    print("\n--- (3) Confirm cuDNN's selected algorithm via CUDNN_LOGINFO_DBG ---")
    env = dict(os.environ)
    env["CUDNN_LOGINFO_DBG"] = "1"
    env["CUDNN_LOGDEST_DBG"] = "stdout"
    # Some cuDNN versions key on the legacy/level variables; set both for portability.
    env["CUDNN_LOGLEVEL_DBG"] = "3"
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--cudnn-log-worker",
           "--shape", str(dims["N"]), str(dims["C"]), str(dims["H"]),
           str(dims["W"]), str(dims["C_out"]),
           "--k", str(k), "--stride", str(stride)]
    print("      invoking:  CUDNN_LOGINFO_DBG=1 CUDNN_LOGDEST_DBG=stdout \\\n"
          f"                 {' '.join(cmd[1:])}")
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    except Exception as e:
        print(f"      [warn] subprocess failed: {e}")
        return False, []
    log = out.stdout + out.stderr
    hits = [ln for ln in log.splitlines() if "winograd" in ln.lower()]
    engine_lines = [ln for ln in log.splitlines()
                    if any(t in ln.lower() for t in ("engine", "algo", "winograd"))]
    if hits:
        print(f"      CONFIRMED: cuDNN log mentions Winograd ({len(hits)} line(s)). Sample:")
        for ln in hits[:5]:
            print(f"        {ln.strip()[:140]}")
    elif engine_lines:
        print("      cuDNN debug log captured but no 'winograd' token found. Engine/algo lines:")
        for ln in engine_lines[:5]:
            print(f"        {ln.strip()[:140]}")
    else:
        print("      [info] No cuDNN debug log captured (this cuDNN build may not honor "
              "CUDNN_LOGINFO_DBG, or logging is compiled out). The eligible-vs-ineligible "
              "proxy (measurement 2) does not depend on this step.")
    return bool(hits), hits


def _cudnn_log_worker(ns):
    """Child entry point: run exactly one cuDNN conv so its algo is logged."""
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    N, C, H, W, C_out = ns.shape
    k, stride, pad = ns.k, ns.stride, ns.k // 2
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    w = torch.randn(C_out, C, k, k, device="cuda", dtype=torch.float16)
    for _ in range(3):  # let cuDNN.benchmark run its algorithm search (logs engines)
        y = torch.nn.functional.conv2d(x, w, None, stride, pad, (1, 1), 1)
    torch.cuda.synchronize()
    print(f"[worker] ran conv2d k={k} stride={stride} out={tuple(y.shape)}")


def _free(*ts):
    for t in ts:
        del t
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny shape + few reps for a <60s build-time check")
    ap.add_argument("--cudnn-log", action="store_true",
                    help="only run the cuDNN debug-log algorithm-confirmation step")
    ap.add_argument("--cudnn-log-worker", action="store_true",
                    help=argparse.SUPPRESS)  # internal subprocess entry point
    ap.add_argument("--shape", type=int, nargs=5, metavar=("N", "C", "H", "W", "Cout"),
                    default=None, help="conv input/weight dims; A100/H100 can scale up")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this experiment requires a GPU.")
        sys.exit(2)

    # Internal subprocess mode for the cuDNN log worker.
    if args.cudnn_log_worker:
        _cudnn_log_worker(args)
        return

    dims = (dict(zip(("N", "C", "H", "W", "C_out"), args.shape)) if args.shape
            else (SMOKE if args.smoke else DEFAULT))

    if args.smoke:
        warmup = args.warmup if args.warmup is not None else 3
        reps = args.reps if args.reps is not None else 5
        print("[SMOKE] tiny shape, few reps — plumbing only, not for the paper.")
    else:
        warmup = args.warmup if args.warmup is not None else 15
        reps = args.reps if args.reps is not None else 50

    banner("Experiment 4 — Winograd isolation in the conv gap  (RC4)")
    print(f"  conv input (N,C,H,W)=({dims['N']},{dims['C']},{dims['H']},{dims['W']}) "
          f"weight C_out={dims['C_out']} fp16")

    # --cudnn-log: just the observability step (documented standalone invocation).
    if args.cudnn_log:
        confirm_cudnn_algo(dims, k=args.k, stride=args.stride)
        return

    rows = []
    cudnn_determinism_ab(dims, warmup, reps, rows)
    eligible_vs_ineligible(dims, warmup, reps, rows)
    # Observability step (best-effort; never fatal).
    confirm_cudnn_algo(dims, k=3, stride=1)

    write_csv(EXPERIMENT, rows, [
        "measurement", "config", "impl", "winograd_eligible",
        "median_ms", "mean_ms", "std_ms", "gap_vs_cudnn", "e_vs_cudnn", "note",
    ])


if __name__ == "__main__":
    main()
