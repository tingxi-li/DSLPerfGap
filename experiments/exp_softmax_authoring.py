#!/usr/bin/env python3
"""Experiment — Softmax authoring collapse (RQ1 evaluation gap / RQ2 authoring cause).

The in-tree ViperBench TileLang softmax caches the ENTIRE row in shared memory
(`S = T.alloc_shared((N,))`, see ViperBench/softmax/tilelang_impl.py). At fp16 that
is `N*2` bytes/block. Once it exceeds the A100-SXM4's 48 KB *static* shared-memory
budget (N > 24576) occupancy collapses and the kernel --- still functionally
correct, and fast on the GH200 (sm_90) and at small N --- slows by ~2 orders of
magnitude. The `opt_kernels` streaming variant (block_N=4096 tiles, no full-row
cache) stays flat across all N.

This is a concrete evaluation-gap instance with an *authoring* root cause: a
correct, hand-"optimized" DSL kernel passes correctness gates and small-shape /
other-GPU benchmarks, yet collapses on a different GPU at the large shape because
of one shared-memory allocation choice. The fix is an authoring pattern (stream
tiles instead of caching the whole row), not a language or library change.

Sweeps N at fixed M, timing {in-tree TileLang, opt_kernels streaming, torch} and
recording the per-block shared-memory footprint and whether it exceeds 48 KB.

Usage:
    python exp_softmax_authoring.py            # full sweep (LARGE; minutes, JIT compiles)
    python exp_softmax_authoring.py --smoke    # tiny, plumbing only
"""
import argparse
import os
import sys

os.environ.setdefault("_SOFTMAX_REEXEC", "1")  # stop opt_kernels softmax_opt from re-exec'ing (detaches timing)
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness as H  # noqa: E402

M = 4096
NS_FULL = [8192, 16384, 24576, 32768]   # 16/32/48/64 KB shared @ fp16 -- brackets the 48 KB budget
NS_SMOKE = [4096, 8192]
SHARED_BUDGET_KB = 48.0                  # A100 per-block STATIC shared-memory cap


def _maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ns = NS_SMOKE if args.smoke else NS_FULL
    warmup, reps = (2, 3) if args.smoke else (10, 30)

    print(f"== softmax authoring collapse == {H.device_info()['gpu_name']}  (48 KB shared budget)")
    intree = H.load_impl("softmax", "tilelang").softmax
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_kernels"))
    import softmax_opt  # noqa: E402
    optfn = softmax_opt.run

    rows = []
    for N in ns:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        ref = torch.softmax(x.float(), dim=-1)
        shared_kb = N * 2 / 1024.0

        def _torch():
            return torch.softmax(x, dim=-1)

        y_in = intree(x); err_in = _maxabs(y_in, ref)
        y_op = optfn(x);  err_op = _maxabs(y_op, ref)
        t_in = H.time_kernel(intree, x, warmup=warmup, reps=reps)["median_ms"]
        t_op = H.time_kernel(optfn, x, warmup=warmup, reps=reps)["median_ms"]
        t_th = H.time_kernel(_torch, warmup=warmup, reps=reps)["median_ms"]

        row = dict(
            M=M, N=N,
            shared_mem_kb=round(shared_kb, 1),
            exceeds_48kb=bool(shared_kb > SHARED_BUDGET_KB),
            intree_ms=round(t_in, 4), opt_ms=round(t_op, 4), torch_ms=round(t_th, 4),
            intree_slowdown_x=round(t_in / t_th, 1),
            opt_slowdown_x=round(t_op / t_th, 2),
            intree_correct=bool(err_in < 2e-2), opt_correct=bool(err_op < 2e-2),
            intree_maxabs_err=f"{err_in:.2e}", opt_maxabs_err=f"{err_op:.2e}",
        )
        rows.append(row)
        print(f"  N={N:<6} shared={shared_kb:>4.0f}KB exceeds48={row['exceeds_48kb']!s:<5} "
              f"intree={t_in:>9.3f}ms ({row['intree_slowdown_x']}x) opt={t_op:>7.3f}ms "
              f"torch={t_th:.3f}ms  correct(in/op)={row['intree_correct']}/{row['opt_correct']}")

    out = H.write_csv("softmax_authoring", rows, list(rows[0].keys()))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
