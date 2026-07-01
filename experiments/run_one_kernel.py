#!/usr/bin/env python3
"""
Single-launch profiling target for Nsight Compute (Exp 1).

`ncu` must profile EXACTLY ONE kernel invocation, or it serializes every launch
and a run takes hours.  This harness:

  1. Loads one ViperBench kernel impl  (<kernel> <impl> <size>)  via
     `_harness.load_impl`  (impl in {pytorch, triton, tilelang}).
  2. Resolves the *exact paper shape* by reusing `ViperBench/benchmark.py`'s
     `get_test_cases()` when importable; otherwise it falls back to user-supplied
     shape args so the harness still works on a stripped checkout.
  3. WARMS UP the kernel several times so TileLang/Triton JIT + autotune compile
     is finished BEFORE profiling  (compile must NOT be inside the counted launch).
  4. Runs the target ONCE inside an NVTX range named "TARGET" via
     torch.cuda.nvtx.range_push/pop.

`ncu` then isolates that single launch with:
    ncu --target-processes all --nvtx --nvtx-include "TARGET/" --launch-count 1 ...
(`--target-processes all` is required because Python spawns the CUDA process.)

Portability: nothing here is sm_89-specific.  The SAME script runs on A100/H100;
TileLang/Triton JIT-recompile for the target SM automatically.

Usage:
    python run_one_kernel.py <kernel> <impl> [size]
    python run_one_kernel.py matmul triton large
    python run_one_kernel.py matmul pytorch large        # cuBLAS reference
    python run_one_kernel.py layer_norm tilelang large
    python run_one_kernel.py argmax tilelang large

Options (all optional):
    --warmup N        warmup iterations before the profiled launch (default 5)
    --range NAME      NVTX range name ncu isolates on        (default TARGET)
    --list            list available (size) labels for <kernel> and exit
"""
import argparse
import os
import sys

# Make `_harness` (this dir) and ViperBench importable regardless of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_VIPER_DIR = os.path.join(_REPO_ROOT, "ViperBench")
for _p in (_THIS_DIR, _REPO_ROOT, _VIPER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from _harness import load_impl  # noqa: E402


def _get_cases():
    """Return ViperBench's get_test_cases() dict, or None if not importable.

    Imported lazily so this script still runs on a checkout where benchmark.py
    is absent or has unrelated import errors -- in that case the caller must
    pass an explicit shape.
    """
    try:
        import importlib.util
        bpath = os.path.join(_VIPER_DIR, "benchmark.py")
        if not os.path.exists(bpath):
            return None
        spec = importlib.util.spec_from_file_location("viper_benchmark", bpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_test_cases()
    except Exception as e:  # pragma: no cover - diagnostic path
        print(f"[run_one_kernel] note: could not import get_test_cases(): {e}",
              file=sys.stderr)
        return None


def _load_opt_kernel(kernel):
    """Load experiments/opt_kernels/<kernel>_opt.py (the RQ3 optimized kernel).

    These modules re-exec themselves once at import to fix up LD_LIBRARY_PATH
    unless a sentinel env var is set; that re-exec would detach ncu from the
    profiled process, so we set the sentinels first to keep a single process.
    The module exposes run(x) and get_inputs() (the paper large shape).
    """
    import importlib.util
    os.environ.setdefault("_LSE_REEXEC", "1")
    os.environ.setdefault("_SOFTMAX_REEXEC", "1")
    path = os.path.join(_THIS_DIR, "opt_kernels", f"{kernel}_opt.py")
    if not os.path.exists(path):
        print(f"[run_one_kernel] ERROR: no optimized kernel at {path}", file=sys.stderr)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(f"{kernel}_opt", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for need in ("run", "get_inputs"):
        if not hasattr(m, need):
            print(f"[run_one_kernel] ERROR: {path} has no {need}()", file=sys.stderr)
            sys.exit(1)
    return m


def resolve_case(kernel, size, cases):
    """Find the (size_label, fn_name, args, kwargs, desc, is_slow) tuple.

    The ViperBench tuple layout is exactly the one the protocol relies on:
        next(c for c in get_test_cases()[kernel] if c[0] == size)
    """
    if cases is None or kernel not in cases:
        return None
    for c in cases[kernel]:
        if c[0] == size:
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description="Single-launch ncu profiling target.")
    ap.add_argument("kernel", help="ViperBench kernel dir name, e.g. matmul, conv2d")
    ap.add_argument("impl", choices=["pytorch", "triton", "tilelang", "tilelang_opt"],
                    help="implementation to profile; 'tilelang_opt' loads the RQ3 "
                         "optimized kernel from experiments/opt_kernels/<kernel>_opt.py")
    ap.add_argument("size", nargs="?", default="large",
                    help="size label from get_test_cases (default: large)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="warmup iters before the profiled launch (JIT/autotune)")
    ap.add_argument("--range", default="TARGET", dest="range_name",
                    help='NVTX range name ncu isolates on (default "TARGET")')
    ap.add_argument("--list", action="store_true",
                    help="list available size labels for <kernel> and exit")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[run_one_kernel] ERROR: CUDA not available.", file=sys.stderr)
        return 1

    if args.impl == "tilelang_opt":
        # Optimized kernel from experiments/opt_kernels/<kernel>_opt.py (the RQ3
        # "after" kernel), loaded directly so ncu profiles its lowered launch --
        # e.g. this is the path that captures the logsumexp sm_90 register spill.
        mod = _load_opt_kernel(args.kernel)
        if args.list:
            xs = mod.get_inputs()
            print(f"{args.kernel}_opt get_inputs -> {[tuple(t.shape) for t in xs]}")
            return 0
        fn = mod.run
        fn_args = tuple(mod.get_inputs())
        fn_kwargs = {}
        x0 = fn_args[0]
        size_label = args.size
        desc = (f"opt_kernels/{args.kernel}_opt.py:run() "
                f"{tuple(x0.shape)} {str(x0.dtype).replace('torch.', '')}")
    else:
        cases = _get_cases()

        if args.list:
            if cases and args.kernel in cases:
                print(f"sizes for {args.kernel}:")
                for c in cases[args.kernel]:
                    print(f"  {c[0]:<10} {c[4]}")
            else:
                print(f"[run_one_kernel] no cases found for {args.kernel}")
            return 0

        case = resolve_case(args.kernel, args.size, cases)
        if case is None:
            avail = ""
            if cases and args.kernel in cases:
                avail = " available sizes: " + ", ".join(c[0] for c in cases[args.kernel])
            print(f"[run_one_kernel] ERROR: no test case '{args.size}' for kernel "
                  f"'{args.kernel}'.{avail}\n"
                  f"  (Is ViperBench/benchmark.py importable on this machine?)",
                  file=sys.stderr)
            return 1

        size_label, fn_name, fn_args, fn_kwargs, desc, _is_slow = case
        fn_kwargs = fn_kwargs or {}

        # Load the impl module and resolve the unified-API function (== fn_name).
        mod = load_impl(args.kernel, args.impl)
        if not hasattr(mod, fn_name):
            print(f"[run_one_kernel] ERROR: {args.impl}_impl for {args.kernel} has no "
                  f"function '{fn_name}'. Has: "
                  f"{[a for a in dir(mod) if not a.startswith('_')]}", file=sys.stderr)
            return 1
        fn = getattr(mod, fn_name)

    dev = torch.cuda.get_device_name(0)
    print(f"[run_one_kernel] {args.kernel} / {args.impl} / {size_label}  ({desc})")
    print(f"[run_one_kernel] GPU: {dev}")
    print(f"[run_one_kernel] warmup={args.warmup}  nvtx_range='{args.range_name}'")

    # --- Warmup: triggers TileLang/Triton JIT + autotune so compile is NOT -------
    # --- inside the profiled launch.  Synchronize so all compile work drains. ----
    for _ in range(max(0, args.warmup)):
        fn(*fn_args, **fn_kwargs)
    torch.cuda.synchronize()

    # --- The single profiled launch, fenced by the NVTX "TARGET" range. ----------
    # ncu --nvtx --nvtx-include "TARGET/" --launch-count 1 isolates exactly this.
    torch.cuda.nvtx.range_push(args.range_name)
    out = fn(*fn_args, **fn_kwargs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # Touch the output so dead-code elimination can never drop the launch.
    try:
        if isinstance(out, torch.Tensor):
            _ = float(out.reshape(-1)[0].item()) if out.numel() else 0.0
    except Exception:
        pass

    print(f"[run_one_kernel] done: 1 launch profiled inside '{args.range_name}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
