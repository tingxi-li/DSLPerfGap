"""
exp_significance.py — Path A: re-measure the near-parity comparisons with GPU
clocks LOCKED, reporting median + std-dev + p95 and a significance verdict
(does the efficiency delta exceed run-to-run noise?).

Answers R1's load-bearing rigor critique (reviews.txt:50): a ~9% clock-variation
noise floor means small efficiency gaps "(e.g., 94.6% vs. 97.8%)" may not be
meaningful unless clocks are locked. profile.csv pins those two numbers to
layer_norm (94.5%) and softmax (97.6%); this script re-times the full
noise-sensitive set with dispersion and classifies each as significant-vs-noise.

RUN ONLY WITH CLOCKS LOCKED:
    sudo nvidia-smi -i 0 -pm 1
    sudo nvidia-smi -i 0 -lmc 9001
    sudo nvidia-smi -i 0 -lgc 1400          # fallback 1350 if it won't hold
    python experiments/exp_significance.py
    sudo nvidia-smi -i 0 -rgc -rmc          # reset afterward
The script self-checks the clocks are pinned and ABORTS otherwise, so no
unlocked data is ever recorded. Numbers here are revision material — the
rebuttal text stays qualitative.

Reuses experiments/_harness.py (time_kernel, load_impl, write_csv, banner) and
ViperBench/benchmark.py's get_test_cases() for the exact same input tensors.
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from _harness import (  # noqa: E402
    banner, time_kernel, load_impl, write_csv, library_efficiency,
)

VIPER_DIR = HERE.parent / "ViperBench"

# Near-parity / noise-sensitive kernels (E_lib in [85,115]% in profile.csv, where
# the ~9% clock-noise band can flip "parity vs not"), plus conv2d — the kernel
# whose 9% run-to-run variation R1 cites from Table 7.
KERNELS = ["layer_norm", "softmax", "mean_reduction", "relu", "add",
           "swiglu", "index_select", "cross_entropy", "conv2d"]
SIZE = "large"
IMPLS = ["pytorch", "triton", "tilelang"]

# Locked-clock self-check: accept either the primary 1400 or the 1350 fallback.
GR_OK = lambda g: (1340 <= g <= 1460) or (1290 <= g <= 1410)  # noqa: E731
MEM_MIN = 8900
Z95 = 1.96


def _get_cases():
    """ViperBench get_test_cases() — same input tensors benchmark.py uses."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "viper_benchmark", str(VIPER_DIR / "benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_test_cases()


def resolve_case(kernel, size, cases):
    for c in cases.get(kernel, []):
        if c[0] == size:
            return c
    return None


def query_clocks(idx=0):
    """Read-only: current (graphics, memory) clocks in MHz for GPU idx."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem",
         "--format=csv,noheader,nounits", "-i", str(idx)],
        capture_output=True, text=True, timeout=10)
    gr, mem = out.stdout.strip().splitlines()[0].split(",")
    return int(gr), int(mem)


def assert_clocks_locked(allow_unlocked, idx=0):
    gr, mem = query_clocks(idx)
    ok = GR_OK(gr) and mem >= MEM_MIN
    print(f"  clock self-check: graphics={gr} MHz, memory={mem} MHz  -> "
          f"{'LOCKED ok' if ok else 'NOT pinned'}")
    if not ok and not allow_unlocked:
        print("  ABORT: clocks are not pinned. Lock them first:\n"
              "    sudo nvidia-smi -i 0 -lmc 9001 -lgc 1400\n"
              "  then re-run. (Use --allow-unlocked only for a dev smoke test.)",
              file=sys.stderr)
        sys.exit(2)
    if not ok:
        print("  --allow-unlocked: continuing on UNLOCKED clocks (smoke test only).")
    return gr, mem


def time_arm(label, call, warmup, reps):
    """Explicit warmup (absorbs JIT/autotune) then timed reps. dict|'oom'|'error'."""
    try:
        call()
        torch.cuda.synchronize()
        return time_kernel(call, warmup=warmup, reps=reps)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"      {label:<10} OOM")
        return "oom"
    except Exception as e:
        print(f"      {label:<10} ERROR: {type(e).__name__}: {str(e)[:90]}")
        return "error"


def verdict(e_lib, sigma_rel):
    """Is the DSL's efficiency distinguishable from library parity (100%)?
    band = 95% half-interval on E_lib (percentage points), from propagated std."""
    band = Z95 * sigma_rel * e_lib
    if abs(e_lib - 100.0) <= band:
        return "within noise of parity", round(band, 2)
    return ("significant: DSL slower" if e_lib < 100 else
            "significant: DSL faster"), round(band, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="fast plumbing check: 3 kernels, 20 reps, unlocked OK, "
                         "writes significance_smoke.csv")
    ap.add_argument("--allow-unlocked", action="store_true",
                    help="do not abort if clocks are not pinned (dev only)")
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()

    kernels = KERNELS
    reps = args.reps
    allow_unlocked = args.allow_unlocked
    experiment = "significance"
    if args.smoke:
        kernels = ["layer_norm", "softmax", "add"]
        reps = 20
        allow_unlocked = True
        experiment = "significance_smoke"

    banner("Significance / clock-locked re-measurement (Path A, R1 reviews.txt:50)")
    locked_gr, locked_mem = assert_clocks_locked(allow_unlocked)
    print(f"  config: reps={reps}, warmup=15, kernels={len(kernels)}, "
          f"experiment={experiment}")

    cases = _get_cases()
    rows = []
    for kernel in kernels:
        case = resolve_case(kernel, SIZE, cases)
        if case is None:
            print(f"  [skip] {kernel}: no '{SIZE}' case")
            continue
        _, fn_name, fn_args, fn_kwargs, desc, _ = case
        fn_kwargs = fn_kwargs or {}
        print(f"\n  {kernel} [{SIZE}]  {desc}")

        timings = {}
        for impl in IMPLS:
            try:
                fn = getattr(load_impl(kernel, impl), fn_name)
            except Exception as e:
                print(f"      {impl:<10} no impl ({str(e)[:60]})")
                continue
            r = time_arm(impl, (lambda f=fn: f(*fn_args, **fn_kwargs)), 15, reps)
            timings[impl] = r
            if isinstance(r, dict):
                rel = 100 * r["std_ms"] / r["mean_ms"] if r["mean_ms"] else 0.0
                print(f"      {impl:<10} median={r['median_ms']:.4f}  "
                      f"mean={r['mean_ms']:.4f}+/-{r['std_ms']:.4f}  ({rel:.1f}% rel-std)")

        pt = timings.get("pytorch")
        bands = {}
        for impl in ("triton", "tilelang"):
            r = timings.get(impl)
            if not isinstance(r, dict):
                continue
            row = dict(kernel=kernel, shape=desc, impl=impl,
                       median_ms=r["median_ms"], mean_ms=r["mean_ms"],
                       std_ms=r["std_ms"], p95_ms=r["p95_ms"],
                       e_lib_pct="", ci95_band_pct="", verdict="",
                       locked_gr_mhz=locked_gr, locked_mem_mhz=locked_mem)
            if isinstance(pt, dict) and r["mean_ms"] > 0 and pt["mean_ms"] > 0:
                e = library_efficiency(pt["median_ms"], r["median_ms"])
                sigma_rel = math.sqrt((pt["std_ms"] / pt["mean_ms"]) ** 2 +
                                      (r["std_ms"] / r["mean_ms"]) ** 2)
                v, band = verdict(e, sigma_rel)
                row.update(e_lib_pct=e, ci95_band_pct=band, verdict=v)
                bands[impl] = (e, band)
                print(f"      -> {impl}: E_lib={e}% +/-{band}pp  [{v}]")
            rows.append(row)

        # pairwise distinguishability (R1's "94.6% vs 97.8%" framing)
        if "triton" in bands and "tilelang" in bands:
            (e1, b1), (e2, b2) = bands["triton"], bands["tilelang"]
            gap, comb = abs(e1 - e2), math.sqrt(b1 ** 2 + b2 ** 2)
            tag = "distinguishable" if gap > comb else "within noise of each other"
            print(f"      -> triton vs tilelang: Δ={gap:.1f}pp (±{comb:.1f}) [{tag}]")

    write_csv(experiment, rows, [
        "kernel", "shape", "impl", "median_ms", "mean_ms", "std_ms", "p95_ms",
        "e_lib_pct", "ci95_band_pct", "verdict", "locked_gr_mhz", "locked_mem_mhz"])
    print("\n  done.")


if __name__ == "__main__":
    main()
