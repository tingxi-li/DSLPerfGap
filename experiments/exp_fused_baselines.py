#!/usr/bin/env python3
"""
Experiment 1 — Split baselines for "library efficiency"
==============================================================================

The paper's single "library efficiency" metric mixes denominators: cuBLAS for
GEMM, cuDNN for conv, but plain **eager PyTorch** for element-wise + RMSNorm.
We split the eager-baseline kernels out so the reader
can tell whether the DSL "gap" is a *real* kernel gap or merely a *fusion*
artifact (eager PyTorch launches many tiny kernels; a fused kernel does not).

For each eager-baseline kernel this script times THREE baselines on the SAME
shape and dtype:

  (a) eager   -- the kernel's own pytorch_impl  (what the paper used)
  (b) fused   -- torch.compile(eager_fn, mode="max-autotune"), warmed up so the
                 one-time compile/autotune is NOT included in the timing
  (c) DSL     -- the Triton and TileLang impls (triton_impl / tilelang_impl)

and reports latency (median + mean+/-std) plus two efficiencies:

  e_vs_eager = t_eager / t_impl * 100      (paper's denominator)
  e_vs_fused = t_fused / t_impl * 100      (the fair, fusion-controlled denominator)

If the DSL looks great vs eager but mediocre vs fused, the "gap" was a fusion
artifact, not a DSL deficiency.

Kernels covered: rms_norm, swiglu, softmax, log_softmax, add, relu, leaky_relu.
(add/relu/leaky_relu are the element-wise representatives; leaky_relu is a
matmul+activation in this suite, so its fused baseline exercises GEMM-epilogue
fusion.)

cross_entropy CONTAMINATION (key honesty fix)
---------------------------------------------
ViperBench/cross_entropy/pytorch_impl.py is NOT a vectorized PyTorch baseline:
its `cross_entropy_fwd` is a *pure-Python triple loop* over rows x column-blocks
with per-element `.item()` host syncs (see lines 25-78). Timing the DSL against
that interpreter loop yields an absurd "library efficiency" (the artifact reports
~800,000% for the large shape: profile.csv has cross_entropy-large PyTorch=15908 ms
vs Triton=1.87 ms ~= 8.5e5%). This script therefore times cross_entropy with BOTH:
  * eager_contaminated -- the existing python-loop impl (FLAGGED in output), and
  * fused_F_cross_entropy -- a proper torch.nn.functional.cross_entropy baseline,
and prints the discrepancy so the contamination is unmistakable. The two have
different output semantics (per-block buffer vs scalar mean loss), so their
latencies are reported but NOT presented as a like-for-like efficiency.

Portability
-----------
  * No arch hardcoding; device props come from _harness (queried at runtime).
  * Shapes parameterized (small/large) with Ada-safe (~20 GB) defaults; a note
    that A100/H100 can scale the "large" column up. Per-arm OOM is caught/skipped.
  * torch.compile mode is configurable (--compile-mode); default max-autotune.

Usage
-----
    python exp_fused_baselines.py            # full sweep (orchestrator runs this)
    python exp_fused_baselines.py --smoke    # tiny shapes, <60s, build-time check
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import (  # noqa: E402
    banner, time_kernel, library_efficiency, load_impl, write_csv,
)

EXPERIMENT = "fused_baselines"


# ---------------------------------------------------------------------------
# Input factories.  size in {"small","large","smoke"}.  Shapes are Ada-safe.
# A100/H100 can scale the "large" column up (these are deliberately modest).
# ---------------------------------------------------------------------------
def _t(*shape, dtype=torch.float16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def make_inputs(kernel: str, size: str):
    """Return (args_tuple, dtype) for the eager/fused/DSL call of `kernel`.
    All three baselines for a kernel are fed the identical args."""
    big = {"small": False, "large": True, "smoke": False}[size]
    if kernel == "rms_norm":
        D = 8192 if big else 1024
        R = 8192 if big else 512
        if size == "smoke":
            R, D = 64, 128
        return (_t(R, D), (D,), _t(D)), torch.float16
    if kernel == "swiglu":
        # xy is concatenated [gate|up] along last dim -> output is half width.
        R = 4096 if big else 512
        W = 32768 if big else 8192
        if size == "smoke":
            R, W = 64, 256
        return (_t(R, W),), torch.float16
    if kernel in ("softmax", "log_softmax"):
        R = 4096 if big else 512
        C = 32768 if big else 1024
        if size == "smoke":
            R, C = 64, 256
        return (_t(R, C),), torch.float16
    if kernel == "add":
        N = 64 * 1024 * 1024 if big else 4096
        if size == "smoke":
            N = 4096
        return (_t(N), _t(N)), torch.float16
    if kernel == "relu":
        N = 16384 if big else 4096
        if size == "smoke":
            N = 512
        return (_t(N, N),), torch.float16
    if kernel == "leaky_relu":
        # matmul + activation: leaky_relu(a, b, "leaky_relu"); a:(M,K) b:(K,N)
        N = 8192 if big else 4096
        if size == "smoke":
            N = 512
        return (_t(N, N), _t(N, N), "leaky_relu"), torch.float16
    raise KeyError(kernel)


def make_ce_inputs(size: str):
    """cross_entropy_fwd args (logits fp32, labels int64, + scalar config).
    Matches benchmark.py's signature exactly."""
    if size == "large":
        R, C = 4096, 32768
    elif size == "smoke":
        R, C = 64, 256
    else:
        R, C = 256, 1024
    logits = _t(R, C, dtype=torch.float32)
    labels = torch.randint(0, C, (R,), device="cuda", dtype=torch.int64)
    # smoothing, logit_scale, lse_square_scale, ignored_index,
    # total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING, SPLIT
    args = (logits, labels, 0.0, 1.0, 0.0, -100, C, 0, R, False, False)
    return logits, labels, args


# ---------------------------------------------------------------------------
# Fused (torch.compile) builders.  We compile a closure that captures the
# kernel's static (non-tensor) args so torch.compile only sees tensors and
# fuses cleanly.  Warmed up before timing so compile/autotune is excluded.
# ---------------------------------------------------------------------------
def build_fused(kernel: str, eager_fn, args, compile_mode: str):
    if kernel == "rms_norm":
        normalized_shape = args[1]
        def f(x, w):
            return eager_fn(x, normalized_shape, w)
        compiled = torch.compile(f, mode=compile_mode)
        return lambda a: compiled(a[0], a[2])
    if kernel == "leaky_relu":
        activation = args[2]
        def f(a, b):
            return eager_fn(a, b, activation)
        compiled = torch.compile(f, mode=compile_mode)
        return lambda ar: compiled(ar[0], ar[1])
    if kernel == "add":
        compiled = torch.compile(lambda x, y: eager_fn(x, y), mode=compile_mode)
        return lambda ar: compiled(ar[0], ar[1])
    # single-tensor kernels: swiglu, softmax, log_softmax, relu
    compiled = torch.compile(lambda x: eager_fn(x), mode=compile_mode)
    return lambda ar: compiled(ar[0])


def _call(fn, args):
    """Adapter so every arm is invoked as fn(args_tuple)."""
    return fn(*args)


def load_eager(kernel: str):
    """Load a kernel's pytorch_impl AND register it in sys.modules.

    _harness.load_impl creates the module via spec_from_file_location but does
    not insert it into sys.modules. torch.compile / TorchDynamo needs to import
    the module that owns the traced function's code object (named e.g.
    'rms_norm_pytorch_impl'); without registration it raises
    ModuleNotFoundError. Registering here (in OUR script, not _harness) fixes
    the fused arm without modifying any existing file."""
    mod = load_impl(kernel, "pytorch")
    sys.modules.setdefault(mod.__name__, mod)
    return mod


# ---------------------------------------------------------------------------
# Core timing of one kernel x size across eager/fused/triton/tilelang.
# ---------------------------------------------------------------------------
KERNELS = ["rms_norm", "swiglu", "softmax", "log_softmax", "add", "relu", "leaky_relu"]


def time_arm(label, call, warmup, reps):
    """Warm up (covers torch.compile / JIT) then time. Returns dict or None."""
    try:
        call()                       # extra explicit warmup -> compile not timed
        torch.cuda.synchronize()
        return time_kernel(call, warmup=warmup, reps=reps)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"      {label:<22} OOM")
        return "oom"
    except Exception as e:
        print(f"      {label:<22} ERROR: {type(e).__name__}: {str(e)[:110]}")
        return "error"


def run_kernel(kernel, sizes, warmup, reps, compile_mode, rows):
    eager_mod = load_eager(kernel)
    eager_fn = getattr(eager_mod, kernel)
    triton_fn = getattr(load_impl(kernel, "triton"), kernel)
    try:
        tilelang_fn = getattr(load_impl(kernel, "tilelang"), kernel)
    except Exception as e:
        print(f"  [warn] {kernel}: no tilelang impl ({str(e)[:60]}); skipping that arm")
        tilelang_fn = None

    for size in sizes:
        args, dtype = make_inputs(kernel, size)
        shape = "/".join(str(tuple(a.shape)) for a in args if torch.is_tensor(a))
        print(f"  {kernel} [{size}]  {shape}  ({str(dtype).replace('torch.','')})")

        fused_call = build_fused(kernel, eager_fn, args, compile_mode)
        arms = [
            ("eager",  lambda: _call(eager_fn, args)),
            ("fused",  lambda: fused_call(args)),
            ("triton", lambda: _call(triton_fn, args)),
        ]
        if tilelang_fn is not None:
            arms.append(("tilelang", lambda: _call(tilelang_fn, args)))

        timings = {}
        for label, call in arms:
            r = time_arm(label, call, warmup, reps)
            timings[label] = r
            if isinstance(r, dict):
                print(f"      {label:<22} median={r['median_ms']:.4f} ms  "
                      f"mean={r['mean_ms']:.4f}+/-{r['std_ms']:.4f}")

        t_eager = timings.get("eager")
        t_fused = timings.get("fused")
        for label in [a[0] for a in arms]:
            r = timings[label]
            if isinstance(r, dict):
                e_eager = (library_efficiency(t_eager["median_ms"], r["median_ms"])
                           if isinstance(t_eager, dict) else "")
                e_fused = (library_efficiency(t_fused["median_ms"], r["median_ms"])
                           if isinstance(t_fused, dict) else "")
                rows.append(dict(
                    kernel=kernel, shape=shape, baseline=label,
                    median_ms=r["median_ms"], mean_ms=r["mean_ms"], std_ms=r["std_ms"],
                    e_vs_eager=e_eager, e_vs_fused=e_fused))
            else:
                rows.append(dict(
                    kernel=kernel, shape=shape, baseline=label,
                    median_ms=r, mean_ms=r, std_ms=r, e_vs_eager=r, e_vs_fused=r))


def run_cross_entropy(sizes, warmup, reps, compile_mode, rows):
    """Special case: time the CONTAMINATED python-loop baseline AND a proper
    fused F.cross_entropy, plus the DSL impls. Flag the discrepancy."""
    print("\n  === cross_entropy (CONTAMINATION CHECK) ===")
    eager_fn = getattr(load_eager("cross_entropy"), "cross_entropy_fwd")  # python loop!
    triton_fn = getattr(load_impl("cross_entropy", "triton"), "cross_entropy_fwd")
    try:
        tilelang_fn = getattr(load_impl("cross_entropy", "tilelang"), "cross_entropy_fwd")
    except Exception:
        tilelang_fn = None

    for size in sizes:
        logits, labels, args = make_ce_inputs(size)
        logit_scale = args[3]
        shape = str(tuple(logits.shape))
        print(f"  cross_entropy [{size}]  logits={shape} fp32")

        # Proper fused baseline: standard mean cross-entropy on the scaled logits.
        def fused_ce(lg=logits, lb=labels, s=logit_scale):
            return F.cross_entropy(lg.float() * s, lb)
        fused_ce_c = torch.compile(fused_ce, mode=compile_mode)

        arms = [
            ("eager_contaminated", lambda: eager_fn(*args)),   # NOT compiled (python loop)
            ("fused_F_cross_entropy", lambda: fused_ce_c()),
            ("triton", lambda: triton_fn(*args)),
        ]
        if tilelang_fn is not None:
            arms.append(("tilelang", lambda: tilelang_fn(*args)))

        timings = {}
        # The contaminated python loop is brutally slow at large shapes; cap its reps.
        for label, call in arms:
            w = warmup if label != "eager_contaminated" else min(warmup, 2)
            rp = reps if label != "eager_contaminated" else min(reps, 3)
            r = time_arm(label, call, w, rp)
            timings[label] = r
            if isinstance(r, dict):
                print(f"      {label:<22} median={r['median_ms']:.4f} ms  "
                      f"mean={r['mean_ms']:.4f}+/-{r['std_ms']:.4f}")

        t_eager = timings.get("eager_contaminated")     # contaminated denominator
        t_fused = timings.get("fused_F_cross_entropy")  # honest denominator
        for label in [a[0] for a in arms]:
            r = timings[label]
            if isinstance(r, dict):
                e_eager = (library_efficiency(t_eager["median_ms"], r["median_ms"])
                           if isinstance(t_eager, dict) else "")
                e_fused = (library_efficiency(t_fused["median_ms"], r["median_ms"])
                           if isinstance(t_fused, dict) else "")
                rows.append(dict(
                    kernel="cross_entropy", shape=shape, baseline=label,
                    median_ms=r["median_ms"], mean_ms=r["mean_ms"], std_ms=r["std_ms"],
                    e_vs_eager=e_eager, e_vs_fused=e_fused))
            else:
                rows.append(dict(
                    kernel="cross_entropy", shape=shape, baseline=label,
                    median_ms=r, mean_ms=r, std_ms=r, e_vs_eager=r, e_vs_fused=r))

        # Make the contamination unmistakable in the console.
        if isinstance(t_eager, dict) and isinstance(timings.get("triton"), dict):
            bogus = library_efficiency(t_eager["median_ms"], timings["triton"]["median_ms"])
            print(f"      >> CONTAMINATION: 'library efficiency' of Triton vs the python-loop "
                  f"baseline = {bogus}%  (nonsensical -- baseline is an interpreter loop, "
                  f"not vectorized PyTorch)")
        if isinstance(t_fused, dict) and isinstance(timings.get("triton"), dict):
            honest = library_efficiency(t_fused["median_ms"], timings["triton"]["median_ms"])
            print(f"      >> HONEST: vs fused F.cross_entropy = {honest}%  "
                  f"(note: different output semantics -- per-block buffer vs scalar mean loss; "
                  f"reported for scale, not as a strict like-for-like ratio)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny shapes + few reps + light compile for a <60s check")
    ap.add_argument("--sizes", nargs="+", default=None,
                    choices=["small", "large"],
                    help="which size columns to run (default: small large)")
    ap.add_argument("--kernels", nargs="+", default=None,
                    help="subset of kernels (default: all + cross_entropy)")
    ap.add_argument("--compile-mode", default=None,
                    help="torch.compile mode (default: max-autotune; smoke uses default)")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this experiment requires a GPU.")
        sys.exit(2)

    if args.smoke:
        sizes = args.sizes or ["smoke"]
        # smoke must be <60s: max-autotune compiles are slow, so use the default
        # compile mode (still exercises the fused path / plumbing).
        compile_mode = args.compile_mode or "default"
        warmup = args.warmup if args.warmup is not None else 3
        reps = args.reps if args.reps is not None else 5
        print("[SMOKE] tiny shapes, light compile — plumbing only, not for the paper.")
    else:
        sizes = args.sizes or ["small", "large"]
        compile_mode = args.compile_mode or "max-autotune"
        warmup = args.warmup if args.warmup is not None else 15
        reps = args.reps if args.reps is not None else 50

    banner("Experiment 1 — split baselines (eager / fused / DSL)")
    print(f"  torch.compile mode = {compile_mode}; sizes = {sizes}\n")

    kernels = args.kernels or KERNELS
    rows = []
    for k in kernels:
        if k == "cross_entropy":
            continue
        try:
            run_kernel(k, sizes, warmup, reps, compile_mode, rows)
        except Exception as e:
            print(f"  [error] kernel {k} aborted: {type(e).__name__}: {str(e)[:140]}")

    if (args.kernels is None) or ("cross_entropy" in args.kernels):
        run_cross_entropy(sizes, warmup, reps, compile_mode, rows)

    write_csv(EXPERIMENT, rows, [
        "kernel", "shape", "baseline",
        "median_ms", "mean_ms", "std_ms", "e_vs_eager", "e_vs_fused",
    ])


if __name__ == "__main__":
    main()
