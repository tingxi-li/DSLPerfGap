#!/usr/bin/env python3
"""
Experiment 3 — Autotuned-matmul reconciliation  (answers W7 / R1-Q3 / R2-Q5)
============================================================================

The paper reports two matmul numbers that *look* contradictory:

  * §5  : "heuristic tuning" across all 22 kernels gives Delta = 0pp  (measured
          on the 16384x16384 RQ1 benchmark shape).
  * §7.3: an expanded `@triton.autotune` search (+ GROUP_SIZE_M L2 swizzle)
          gives matmul a 1.66x speedup (measured on 4096x4096).

This script reconciles them by timing, at BOTH shapes, on the SAME GPU:

  (1) cuBLAS                      -- torch.matmul                       (reference)
  (2) plain Triton matmul         -- ViperBench/matmul/triton_impl.py
        = the §5 kernel: a single block-tile config drawn from the heuristic
          `tuning/` grid (no GROUP_SIZE_M, no num_warps/num_stages search).
  (3) expanded-autotune Triton    -- AKO4ALL/results/optimized/matmul_triton.py
        = the §7.3 kernel: 12-config `@triton.autotune` + GROUP_SIZE_M swizzle.

Expected story (to be confirmed by the orchestrator's full run):
  * At 4096^2 the expanded autotune wins (~1.66x over the plain kernel) — this is
    where §7.3 measured.
  * At 16384^2 the plain kernel's heuristic-tuned config is already close to
    optimal, so the expanded search gives ~0 extra gain — reconciling §5's Delta=0pp.

Note on TileLang: the ViperBench TileLang matmul ALREADY uses `T.use_swizzle`
(ViperBench/matmul/tilelang_impl.py:29), so the swizzle lever §7.3 adds to Triton
is not a fresh TileLang contribution. We time it too for completeness/context.

Portability
-----------
  * Nothing is hardcoded to sm_89. Device props are queried via _harness.
  * Shapes are parameterized (--shapes). Ada-safe defaults (4096, 16384); the
    16384^2 fp16 working set is A+B+C = 3 * 16384^2 * 2B = 1.5 GB, well within
    ~20 GB. On A100/H100 you can add larger shapes, e.g.
        python exp_autotune_matmul.py --shapes 4096 16384 32768
  * Per-shape OOM is caught and the shape is skipped (recorded as oom) so one
    big shape never aborts the sweep.

Usage
-----
    python exp_autotune_matmul.py            # full timing sweep (orchestrator)
    python exp_autotune_matmul.py --smoke    # tiny shapes, <60s, build-time check
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import (  # noqa: E402
    banner, device_info, time_kernel, library_efficiency,
    load_impl, load_optimized, write_csv,
)

EXPERIMENT = "autotune_matmul"

# Ada-safe defaults. 16384^2 fp16 ~= 1.5 GB working set (fits ~20 GB).
# A100/H100 can scale up by passing extra --shapes (e.g. 32768).
DEFAULT_SHAPES = [4096, 16384]
SMOKE_SHAPES = [256, 512]


def _make_square(n: int, dtype=torch.float16):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    return a, b


def _free(*ts):
    for t in ts:
        del t
    torch.cuda.empty_cache()


def run(shapes, warmup, reps):
    info = banner("Experiment 3 — autotuned-matmul reconciliation (W7)")

    # Load the three matmul implementations once (modules cache their JIT).
    plain = load_impl("matmul", "triton")          # §5 heuristic-tuned single config
    cublas = load_impl("matmul", "pytorch")        # cuBLAS via torch.matmul
    expanded = load_optimized("matmul_triton")     # §7.3 @triton.autotune + GROUP_SIZE_M
    try:
        tilelang_mm = load_impl("matmul", "tilelang")  # already uses T.use_swizzle
        have_tilelang = True
    except Exception as e:
        print(f"  [warn] TileLang matmul unavailable, skipping that arm: {e}")
        tilelang_mm = None
        have_tilelang = False

    # (label, callable factory). The factory binds a/b at call time.
    impls = [
        ("cublas", lambda a, b: cublas.matmul(a, b)),
        ("triton_plain", lambda a, b: plain.matmul(a, b)),
        ("triton_autotune", lambda a, b: expanded.matmul(a, b)),
    ]
    if have_tilelang:
        impls.append(("tilelang_swizzle", lambda a, b: tilelang_mm.matmul(a, b)))

    rows = []
    for n in shapes:
        print(f"\n--- shape {n}x{n} (fp16, working set ~= {3 * n * n * 2 / 1e9:.2f} GB) ---")
        try:
            a, b = _make_square(n)
        except torch.cuda.OutOfMemoryError:
            print(f"  [skip] OOM allocating {n}x{n} inputs")
            rows.append(dict(shape=f"{n}x{n}", impl="-", M=n, N=n, K=n,
                             median_ms="oom", mean_ms="oom", std_ms="oom",
                             gflops="oom", speedup_vs_plain="oom", e_vs_cublas="oom"))
            torch.cuda.empty_cache()
            continue

        flops = 2.0 * n * n * n  # multiply-add counted as 2 FLOP
        timings = {}
        for label, fn in impls:
            try:
                # Warm up: this triggers Triton autotune / TileLang JIT so the
                # one-time autotune sweep & compile are NOT included in timing.
                ref = fn(a, b)
                torch.cuda.synchronize()
                t = time_kernel(fn, a, b, warmup=warmup, reps=reps)
                gflops = flops / (t["median_ms"] * 1e-3) / 1e9
                timings[label] = t["median_ms"]
                rows.append(dict(
                    shape=f"{n}x{n}", impl=label, M=n, N=n, K=n,
                    median_ms=t["median_ms"], mean_ms=t["mean_ms"], std_ms=t["std_ms"],
                    gflops=round(gflops, 1),
                    speedup_vs_plain="",   # filled below once plain is known
                    e_vs_cublas="",        # filled below once cublas is known
                ))
                print(f"  {label:<18} median={t['median_ms']:.4f} ms  "
                      f"mean={t['mean_ms']:.4f}+/-{t['std_ms']:.4f}  {gflops:8.1f} GFLOP/s")
                del ref
            except torch.cuda.OutOfMemoryError:
                print(f"  {label:<18} OOM")
                rows.append(dict(shape=f"{n}x{n}", impl=label, M=n, N=n, K=n,
                                 median_ms="oom", mean_ms="oom", std_ms="oom",
                                 gflops="oom", speedup_vs_plain="oom", e_vs_cublas="oom"))
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {label:<18} ERROR: {type(e).__name__}: {str(e)[:120]}")
                rows.append(dict(shape=f"{n}x{n}", impl=label, M=n, N=n, K=n,
                                 median_ms="error", mean_ms="error", std_ms="error",
                                 gflops="error", speedup_vs_plain="error", e_vs_cublas="error"))

        # Fill in the two derived columns for this shape's rows.
        t_plain = timings.get("triton_plain")
        t_cublas = timings.get("cublas")
        for r in rows:
            if r["shape"] != f"{n}x{n}" or r["median_ms"] in ("oom", "error"):
                continue
            if t_plain:
                r["speedup_vs_plain"] = round(t_plain / r["median_ms"], 3)
            if t_cublas is not None:
                # E_lib = t_library / t_DSL * 100  (paper's primary metric).
                r["e_vs_cublas"] = library_efficiency(t_cublas, r["median_ms"])

        # The headline reconciliation line for this shape.
        if "triton_plain" in timings and "triton_autotune" in timings:
            gain = timings["triton_plain"] / timings["triton_autotune"]
            print(f"  >> autotune vs plain @ {n}^2: {gain:.3f}x "
                  f"({'§7.3 1.66x regime' if n <= 8192 else 'Delta~=0pp regime (§5)'})")

        _free(a, b)

    write_csv(EXPERIMENT, rows, [
        "shape", "impl", "M", "N", "K",
        "median_ms", "mean_ms", "std_ms", "gflops",
        "speedup_vs_plain", "e_vs_cublas",
    ])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny shapes + few reps for a <60s build-time check")
    ap.add_argument("--shapes", type=int, nargs="+", default=None,
                    help="square matmul dims (e.g. 4096 16384 32768). "
                         "Defaults: 4096 16384 (Ada-safe).")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this experiment requires a GPU.")
        sys.exit(2)

    if args.smoke:
        shapes = args.shapes or SMOKE_SHAPES
        warmup = args.warmup if args.warmup is not None else 3
        reps = args.reps if args.reps is not None else 5
        print("[SMOKE] tiny shapes, few reps — correctness/plumbing only, not for the paper.")
    else:
        shapes = args.shapes or DEFAULT_SHAPES
        warmup = args.warmup if args.warmup is not None else 15
        reps = args.reps if args.reps is not None else 50

    run(shapes, warmup, reps)


if __name__ == "__main__":
    main()
