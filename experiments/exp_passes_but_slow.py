#!/usr/bin/env python3
"""
Experiment — "Passes-but-slow" demonstration  (RQ1 headline evidence)
=====================================================================

THE CLAIM:

    A *naive but functionally-correct* DSL kernel PASSES a KernelBench-style
    correctness gate yet is many times SLOWER than the vendor/PyTorch library.
    This proves the benchmark's performance gate is weak/absent: nothing in the
    accept/reject protocol turns "slow" into "fail".

What makes this airtight (and reviewer-fair):

  * The gate is the REAL, vendored KernelBench-style evaluator in this repo, not
    a mock. We call its core entry point directly:

        eval_kernel_against_ref(...)   AKO4ALL/bench/kernelbench/bench.py:491

    whose ACCEPT/REJECT decision is, verbatim (bench.py:1009):

        sys.exit(0 if result.correctness else 1)

    i.e. the process exits 0 (ACCEPTED) on correctness ALONE, independent of
    runtime. We invoke the gate with measure_performance=False, which mirrors
    that decision exactly: only `result.correctness` gates the verdict. The
    one performance-adjacent guard the gate has — the >10x "excessive speedup"
    check (bench.py:725-734) — is warn-only AND only ever fires on suspiciously
    *fast* kernels, so a slow-but-correct kernel produces no warning whatsoever.
    We surface that explicitly in the `reward_hack_warning` column.

  * The kernels are the genuine round-1 naive ViperBench TileLang impls
    (`T.serial` reduction loops), exported through the SAME bridge AKO4ALL uses
    (`prepare_kernel.py` -> KERNEL_CONFIGS + generate_kb_wrapper), so the gate
    sees byte-for-byte what a developer would actually submit. We do NOT touch
    AKO4ALL's git-branch optimization workflow (TASK.md); we import its core
    eval function and source-transform helpers directly.

  * Timing uses the house `_harness.time_kernel` (CUDA events, median over reps)
    so dsl_ms / library_ms / slowdown_x are reported the same way as every other
    experiment in experiments/results/<gpu_slug>/.

Demo kernels (most dramatic naive exemplars that are ALSO in KERNEL_CONFIGS;
slowdowns quoted from CLAUDE.md round-1 campaigns are the motivation, measured
fresh here):
    layer_norm  tilelang  bf16   (~1224x slow round-1)
    rms_norm    tilelang  fp16   (~796x  slow round-1)
    argmax      tilelang  fp16   (~9.3x  slow round-1)

Output:  experiments/results/<gpu_slug>/passes_but_slow.csv
Columns: kernel, dsl, dtype, shape, gate_compiled, gate_correct,
         gate_exit_meaning, gate_correct_trials, dsl_ms, library_ms,
         slowdown_x, passed_gate_despite_slow, reward_hack_warning, note

Usage
-----
    python exp_passes_but_slow.py            # full sweep (LARGE shapes; ~minutes)
    python exp_passes_but_slow.py --smoke    # tiny shapes/few reps, plumbing check
    python exp_passes_but_slow.py --kernels layer_norm,argmax   # subset
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import banner, time_kernel, load_impl, write_csv, device_info  # noqa: E402

EXPERIMENT = "passes_but_slow"

REPO_ROOT = Path(__file__).resolve().parent.parent
AKO_DIR = REPO_ROOT / "AKO4ALL"
BENCH_PY = AKO_DIR / "bench" / "kernelbench" / "bench.py"
PREPARE_PY = AKO_DIR / "prepare_kernel.py"


# ---------------------------------------------------------------------------
# Import the REAL gate + the REAL ViperBench->KernelBench export machinery.
# We load them by file path (not as packages) so we depend on no installed
# AKO4ALL package and trigger none of its git/branch workflow.
# ---------------------------------------------------------------------------
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass introspection (which looks the module up
    # in sys.modules via cls.__module__) works for KernelExecResult in bench.py.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_bench = _load_module("kb_bench", BENCH_PY)            # the gate
_prep = _load_module("ako_prepare_kernel", PREPARE_PY)  # KERNEL_CONFIGS + wrapper

# Pull the gate API and the export helpers we use, by their real names.
eval_kernel_against_ref = _bench.eval_kernel_against_ref     # bench.py:491
prepare_solution_source = _bench.prepare_solution_source     # bench.py:785
KERNEL_CONFIGS = _prep.KERNEL_CONFIGS                        # prepare_kernel.py:25
generate_kb_wrapper = _prep.generate_kb_wrapper             # prepare_kernel.py:252
read_impl_source = _prep.read_impl_source                   # prepare_kernel.py:237

PRECISION_TO_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
PRECISION_TO_LABEL = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}


# ---------------------------------------------------------------------------
# Demo set. Each entry is a naive `T.serial` TileLang kernel that ALSO has an
# entry in prepare_kernel.KERNEL_CONFIGS (so the LARGE shape + precision + fn
# name come straight from the real bridge — single source of truth). The only
# thing defined here is a SMALLER `get_inputs` for --smoke plumbing checks.
# ---------------------------------------------------------------------------
DEMOS = [
    {
        "kernel": "layer_norm", "dsl": "tilelang",
        "large_label": "8192x8192",
        "smoke_get_inputs": (
            "def get_inputs():\n"
            "    x = torch.randn(128, 256, device='cuda', dtype=torch.bfloat16)\n"
            "    weight = torch.randn(256, device='cuda', dtype=torch.bfloat16)\n"
            "    bias = torch.randn(256, device='cuda', dtype=torch.bfloat16)\n"
            "    return [x, weight, bias]\n"
        ),
        "smoke_label": "128x256",
    },
    {
        "kernel": "rms_norm", "dsl": "tilelang",
        "large_label": "8192x8192",
        "smoke_get_inputs": (
            "def get_inputs():\n"
            "    x = torch.randn(128, 256, device='cuda', dtype=torch.float16)\n"
            "    normalized_shape = (256,)\n"
            "    weight = torch.randn(256, device='cuda', dtype=torch.float16)\n"
            "    return [x, normalized_shape, weight]\n"
        ),
        "smoke_label": "128x256",
    },
    {
        "kernel": "argmax", "dsl": "tilelang",
        "large_label": "8192x32768 dim=1",
        "smoke_get_inputs": (
            "def get_inputs():\n"
            "    return [torch.randn(128, 512, device='cuda', dtype=torch.float16), 1]\n"
        ),
        "smoke_label": "128x512 dim=1",
    },
]


def _build_inputs(get_inputs_src: str):
    """exec a KernelBench-style get_inputs() source and return its input list.

    Identical to how the gate materializes inputs internally — we reuse the
    exact same source string for the gate AND for our timing so correctness and
    timing are measured on the same shapes.
    """
    ns = {"torch": torch}
    exec(get_inputs_src, ns)
    return ns["get_inputs"]()


def run_gate(kernel: str, dsl: str, get_inputs_src: str, precision: str,
             num_correct_trials: int, verbose: bool = False) -> dict:
    """Run the candidate through the REAL KernelBench-style correctness gate.

    Builds the reference (PyTorch) and solution (naive DSL) KernelBench wrappers
    via the same `prepare_kernel` machinery AKO4ALL uses, applies the gate's own
    `prepare_solution_source` (rename Model->ModelNew, graft the reference tail),
    then calls `eval_kernel_against_ref(measure_performance=False)` — i.e. the
    pure correctness decision that drives the gate's exit code (bench.py:1009).
    Returns the gate verdict (no timing here; timing is done by `time_pair`).
    """
    config = dict(KERNEL_CONFIGS[kernel])
    config["get_inputs"] = get_inputs_src  # swap in the chosen shape (large or smoke)

    ref_src = generate_kb_wrapper(read_impl_source(kernel, "pytorch"), config)
    sol_src = generate_kb_wrapper(read_impl_source(kernel, dsl), config)
    modified_sol = prepare_solution_source(ref_src, sol_src)

    backend = dsl if dsl in ("triton", "tilelang", "cute") else "cuda"
    result = eval_kernel_against_ref(
        original_model_src=ref_src,
        custom_model_src=modified_sol,
        num_correct_trials=num_correct_trials,
        measure_performance=False,     # <-- mirror the exit-code decision: correctness only
        verbose=verbose,
        backend=backend,
        precision=PRECISION_TO_DTYPE[precision],
    )

    if result is None:
        return dict(compiled=False, correct=False, trials="(0 / %d)" % num_correct_trials,
                    note="gate returned None (lock/compile retry path)")

    correct = bool(result.correctness)
    return dict(
        compiled=bool(result.compiled),
        correct=correct,
        trials=result.metadata.get("correctness_trials", f"({int(correct)*num_correct_trials} / {num_correct_trials})"),
        note=("" if correct else
              result.metadata.get("correctness_issue")
              or result.metadata.get("runtime_error_name")
              or result.metadata.get("compilation_error_name") or "incorrect"),
    )


def time_pair(kernel: str, dsl: str, get_inputs_src: str, fn_name: str,
              warmup: int, reps: int) -> tuple:
    """Time the naive DSL kernel and the PyTorch/library baseline at the chosen
    shape using the house CUDA-event timer (_harness.time_kernel). Returns
    (dsl_median_ms, library_median_ms)."""
    dsl_fn = getattr(load_impl(kernel, dsl), fn_name)
    lib_fn = getattr(load_impl(kernel, "pytorch"), fn_name)

    dsl_inputs = _build_inputs(get_inputs_src)
    dsl_t = time_kernel(dsl_fn, *dsl_inputs, warmup=warmup, reps=reps)
    # Fresh inputs for the library (same shapes; avoids any cache aliasing).
    lib_inputs = _build_inputs(get_inputs_src)
    lib_t = time_kernel(lib_fn, *lib_inputs, warmup=warmup, reps=reps)
    return dsl_t["median_ms"], lib_t["median_ms"]


def run_demo(demo: dict, smoke: bool, warmup: int, reps: int,
             num_correct_trials: int, verbose: bool) -> dict:
    kernel, dsl = demo["kernel"], demo["dsl"]
    config = KERNEL_CONFIGS[kernel]
    fn_name = config["fn"]
    precision = config["precision"]
    dtype_label = PRECISION_TO_LABEL[precision]

    if smoke:
        get_inputs_src = demo["smoke_get_inputs"]
        shape = demo["smoke_label"]
    else:
        get_inputs_src = config["get_inputs"]      # the real LARGE shape from the bridge
        shape = demo["large_label"]

    print(f"\n--- {kernel} ({dsl}, {dtype_label}) @ {shape} ---")

    # (a) the REAL correctness gate -> PASS/FAIL + verdict string.
    gate = run_gate(kernel, dsl, get_inputs_src, precision, num_correct_trials, verbose)
    gate_exit_meaning = (
        "exit 0 = ACCEPTED (gate keys on correctness only; bench.py:1009)"
        if gate["correct"] else
        "exit 1 = REJECTED (incorrect)"
    )
    print(f"    gate: compiled={gate['compiled']} correct={gate['correct']} "
          f"trials={gate['trials']}  ->  {gate_exit_meaning}")
    if not gate["correct"] and gate["note"]:
        print(f"    gate note: {gate['note']}")

    # (b) time the naive DSL kernel vs the library baseline at this shape.
    dsl_ms = library_ms = float("nan")
    timing_note = ""
    try:
        dsl_ms, library_ms = time_pair(kernel, dsl, get_inputs_src, fn_name, warmup, reps)
    except Exception as e:  # pragma: no cover - defensive; record but keep sweeping
        timing_note = f"timing_error: {type(e).__name__}: {str(e)[:80]}"
        print(f"    [warn] {timing_note}")

    # (c) slowdown + the headline boolean.
    if library_ms and library_ms == library_ms and library_ms > 0 \
            and dsl_ms == dsl_ms:  # not NaN
        slowdown_x = round(dsl_ms / library_ms, 2)
        eff_speedup = library_ms / dsl_ms   # the gate's >10x guard looks at this
    else:
        slowdown_x = float("nan")
        eff_speedup = float("nan")

    passed_gate_despite_slow = bool(gate["correct"] and slowdown_x == slowdown_x
                                    and slowdown_x > 1.0)

    # The >10x reward-hacking guard (bench.py:725-734) is warn-only AND only
    # triggers when effective_speedup > 10 (a suspiciously FAST kernel). A slow
    # kernel has effective_speedup < 1, so it is *never* flagged.
    if eff_speedup == eff_speedup:  # not NaN
        reward_hack_warning = (
            f"none (effective_speedup={eff_speedup:.3f}x <= 10x guard; "
            f"warn-only guard fires only on suspiciously-fast kernels)"
        )
    else:
        reward_hack_warning = "n/a (timing unavailable)"

    if dsl_ms == dsl_ms and library_ms == library_ms:
        print(f"    timing: dsl={dsl_ms:.4f} ms  library={library_ms:.4f} ms  "
              f"-> {slowdown_x:.2f}x slower")
    print(f"    VERDICT: passes_gate_despite_slow = {passed_gate_despite_slow}")

    return dict(
        kernel=kernel, dsl=dsl, dtype=dtype_label, shape=shape,
        gate_compiled=gate["compiled"], gate_correct=gate["correct"],
        gate_exit_meaning=gate_exit_meaning, gate_correct_trials=gate["trials"],
        dsl_ms=round(dsl_ms, 5) if dsl_ms == dsl_ms else "",
        library_ms=round(library_ms, 5) if library_ms == library_ms else "",
        slowdown_x=slowdown_x if slowdown_x == slowdown_x else "",
        passed_gate_despite_slow=passed_gate_despite_slow,
        reward_hack_warning=reward_hack_warning,
        note=(timing_note or gate["note"]),
    )


def _print_summary(rows: list):
    print("\n" + "=" * 96)
    print("  SUMMARY — 'passes-but-slow': naive DSL kernels that PASS the gate yet are far slower")
    print("=" * 96)
    hdr = (f"  {'kernel':<13}{'dsl':<10}{'dtype':<6}{'shape':<18}"
           f"{'gate':<7}{'dsl_ms':>12}{'lib_ms':>12}{'slowdown':>11}{'passed?':>9}")
    print(hdr)
    print("  " + "-" * 94)
    for r in rows:
        gate = "PASS" if r["gate_correct"] else "FAIL"
        dsl_ms = f"{r['dsl_ms']:.4f}" if isinstance(r["dsl_ms"], (int, float)) else "-"
        lib_ms = f"{r['library_ms']:.4f}" if isinstance(r["library_ms"], (int, float)) else "-"
        sl = f"{r['slowdown_x']:.2f}x" if isinstance(r["slowdown_x"], (int, float)) else "-"
        print(f"  {r['kernel']:<13}{r['dsl']:<10}{r['dtype']:<6}{r['shape']:<18}"
              f"{gate:<7}{dsl_ms:>12}{lib_ms:>12}{sl:>11}"
              f"{str(r['passed_gate_despite_slow']):>9}")
    n_pass_slow = sum(1 for r in rows if r["passed_gate_despite_slow"])
    print("  " + "-" * 94)
    print(f"  {n_pass_slow}/{len(rows)} naive kernels PASS the KernelBench-style correctness gate "
          f"while being >1x slower than the library.")
    print("  => The gate's accept/reject decision is correctness-only; 'slow' is never 'fail'.")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny shapes + few reps for a fast plumbing check (not for the paper)")
    ap.add_argument("--kernels", type=str, default=None,
                    help="comma-separated subset of {layer_norm,rms_norm,argmax}")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None,
                    help="gate correctness trials (default 5 full / 3 smoke; KernelBench CLI default is 5)")
    ap.add_argument("--verbose", action="store_true", help="verbose gate output")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this experiment requires a GPU.")
        sys.exit(2)

    if args.smoke:
        warmup = args.warmup if args.warmup is not None else 3
        reps = args.reps if args.reps is not None else 5
        trials = args.trials if args.trials is not None else 3
        print("[SMOKE] tiny shapes, few reps — plumbing only, NOT for the paper.")
    else:
        warmup = args.warmup if args.warmup is not None else 15
        reps = args.reps if args.reps is not None else 50
        trials = args.trials if args.trials is not None else 5

    demos = DEMOS
    if args.kernels:
        wanted = {k.strip() for k in args.kernels.split(",")}
        demos = [d for d in DEMOS if d["kernel"] in wanted]
        if not demos:
            print(f"No demo kernels match {sorted(wanted)}; "
                  f"available: {[d['kernel'] for d in DEMOS]}")
            sys.exit(2)

    banner("Passes-but-slow demonstration (RQ1) — naive DSL kernels vs the KernelBench gate")
    print(f"  gate: eval_kernel_against_ref  ({BENCH_PY.relative_to(REPO_ROOT)}:491), "
          f"decision = sys.exit(0 if correctness else 1)  (:1009)")
    print(f"  correctness trials = {trials}  |  timing warmup/reps = {warmup}/{reps}")

    rows = []
    for demo in demos:
        try:
            rows.append(run_demo(demo, args.smoke, warmup, reps, trials, args.verbose))
        except Exception as e:  # pragma: no cover - keep sweeping on a single failure
            print(f"  [error] {demo['kernel']} ({demo['dsl']}): "
                  f"{type(e).__name__}: {str(e)[:120]}")
            rows.append(dict(
                kernel=demo["kernel"], dsl=demo["dsl"],
                dtype=PRECISION_TO_LABEL[KERNEL_CONFIGS[demo["kernel"]]["precision"]],
                shape=(demo["smoke_label"] if args.smoke else demo["large_label"]),
                gate_compiled="", gate_correct="", gate_exit_meaning="ERROR",
                gate_correct_trials="", dsl_ms="", library_ms="", slowdown_x="",
                passed_gate_despite_slow="", reward_hack_warning="",
                note=f"{type(e).__name__}: {str(e)[:120]}"))

    _print_summary(rows)

    write_csv(EXPERIMENT, rows, [
        "kernel", "dsl", "dtype", "shape",
        "gate_compiled", "gate_correct", "gate_exit_meaning", "gate_correct_trials",
        "dsl_ms", "library_ms", "slowdown_x", "passed_gate_despite_slow",
        "reward_hack_warning", "note",
    ])


if __name__ == "__main__":
    main()
