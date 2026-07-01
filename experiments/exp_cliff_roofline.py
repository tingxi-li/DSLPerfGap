#!/usr/bin/env python3
"""
Experiment — Dual-variant cliff + roofline anchor   (RQ2 gap / RQ3 heuristics)
==============================================================================

For every kernel that has a genuine *naive -> optimized* story in this artifact
we time THREE implementations at the large benchmark shape:

    NAIVE      the as-shipped, functionally-correct-but-slow DSL kernel
    OPTIMIZED  the post-mitigation DSL kernel (same DSL; no language switch)
    LIBRARY    the PyTorch / vendor-library reference (cuDNN/cuBLAS/eager)

and emit ONE CSV row per kernel with:

    E_lib_naive = t_lib / t_naive   (ratio; >1 => DSL already beats the library)
    E_lib_opt   = t_lib / t_opt     (ratio; ~1 or >1 => optimized reaches parity)
    cliff       = t_naive / t_opt    (the headline naive->optimized speedup)
    roofline_frac_opt = achieved(opt) / GPU_peak   (baseline-INDEPENDENT anchor)

WHY THIS EXPERIMENT (RQ3 heuristics)
---------------------------------------------------------------------
The comparability rules (within-epsilon-of / at-least-parity-with the library)
are circular: PyTorch is both the baseline AND the de-facto "good" answer. The
*roofline fraction* is the non-circular leg -- it asks "what fraction of the
hardware's memory/compute roofline does the optimized kernel reach?", which is
true regardless of what the library does. The cliff (naive/optimized) quantifies
how badly a correct-but-naive kernel is leaving on the table; pairing it with the
roofline frac shows the optimized kernel is genuinely well-written, not merely
"faster than the naive one".

PROVENANCE (see experiments/cliff/PROVENANCE.md for the exact commit+path of
every naive/optimized file). Briefly:
  * Round-1 kernels (layer_norm/rms_norm/argmax tilelang; matmul/conv2d triton):
      naive  = experiments/cliff/naive/<k>_<dsl>.py   (== current ViperBench impl,
               introduced 7bde14b, never optimized in place)
      opt    = experiments/cliff/opt/<k>_<dsl>.py      (== AKO4ALL/results/optimized)
  * Round-2 kernels (max_reduction tilelang+triton, mean_reduction/softmax/
    log_softmax/batched_matmul/conv2d tilelang):
      naive  = experiments/cliff/naive/<k>_<dsl>.py    (== ViperBench @ fdf6b6e~1
               = a72e84b, the pre-in-place-optimization state)
      opt    = experiments/cliff/opt/<k>_<dsl>.py       (== current ViperBench impl,
               optimized in place by fdf6b6e)

Shapes come from AKO4ALL/prepare_kernel.py KERNEL_CONFIGS (single source of truth).
ONE documented deviation: conv2d is timed with padding=1 (the canonical conv-large
of ViperBench/benchmark.py:129), because KERNEL_CONFIGS' bare get_inputs leaves
padding=0 (OW=126) which does NOT trigger the optimized TileLang conv fast path
(gated on OW%16==0 and OW<=128) -- padding=1 gives OW=128 and matches the paper.

CAVEAT worth reading (surfaced in the `note` column too): the NAIVE softmax falls
back to `torch.softmax` for N>8192, so at the large shape (N=32768) its "naive"
path is a PyTorch call with an fp32 upcast, NOT a TileLang kernel. The cliff is
still a valid naive-impl-vs-optimized-impl number, but it is not a "slow T.serial
DSL kernel" cliff for softmax specifically.

Usage
-----
    python exp_cliff_roofline.py                 # full sweep (LARGE shapes; minutes)
    python exp_cliff_roofline.py --smoke         # tiny shapes/few reps, plumbing only
    python exp_cliff_roofline.py --kernels matmul,layer_norm
    python exp_cliff_roofline.py --kernels conv2d_tilelang,max_reduction_triton

Output:  experiments/results/<gpu_slug>/cliff_roofline.csv
Columns: kernel, dsl, dtype, shape, t_lib_ms, t_naive_ms, t_opt_ms,
         E_lib_naive, E_lib_opt, cliff,
         roofline_bound_type, roofline_peak_assumed, roofline_frac_opt, note

DO NOT RUN ON A SHARED/BUSY GPU: timing medians are corrupted by contention.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import banner, time_kernel, load_impl, write_csv, device_slug  # noqa: E402

EXPERIMENT = "cliff_roofline"

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
CLIFF_DIR = THIS_DIR / "cliff"
NAIVE_DIR = CLIFF_DIR / "naive"
OPT_DIR = CLIFF_DIR / "opt"
PREPARE_PY = REPO_ROOT / "AKO4ALL" / "prepare_kernel.py"

PRECISION_TO_LABEL = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}


# ---------------------------------------------------------------------------
# KERNEL_CONFIGS (shapes/fn/precision) -- single source of truth, by file path.
# prepare_kernel.py imports no torch, so this is GPU-free to load.
# ---------------------------------------------------------------------------
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_prep = _load_module("ako_prepare_kernel_cliff", PREPARE_PY)
KERNEL_CONFIGS = _prep.KERNEL_CONFIGS


# ---------------------------------------------------------------------------
# The cliff set: 12 (kernel, dsl) pairs with a committed naive->optimized story.
#   round    : 1 (AKO4ALL out-of-tree opt) or 2 (fdf6b6e in-place opt)
#   bound    : "memory" or "compute"  (roofline classification; justified below)
#   extra    : positional args appended after get_inputs() (conv2d -> bias,stride,pad)
#   smoke    : a small get_inputs() source string for --smoke plumbing checks
# fn / precision / large get_inputs come from KERNEL_CONFIGS[kernel].
# ---------------------------------------------------------------------------
def _gi(body: str) -> str:
    return "def get_inputs():\n" + body


CLIFF_SET = [
    # ---- Round 1 (naive = current ViperBench; opt = AKO4ALL/results/optimized) ----
    dict(kernel="layer_norm", dsl="tilelang", round=1, bound="memory",
         shape="x:(8192,8192) w,b:(8192) bf16", extra=(),
         smoke=_gi("    x = torch.randn(128, 256, device='cuda', dtype=torch.bfloat16)\n"
                   "    return [x, torch.randn(256, device='cuda', dtype=torch.bfloat16),\n"
                   "            torch.randn(256, device='cuda', dtype=torch.bfloat16)]\n"),
         note="reduction kernel; naive=T.serial mean/var + fp32 I/O + torch.zeros, opt=T.reduce + bf16 I/O"),
    dict(kernel="rms_norm", dsl="tilelang", round=1, bound="memory",
         shape="x:(8192,8192) w:(8192) fp16", extra=(),
         smoke=_gi("    x = torch.randn(128, 256, device='cuda', dtype=torch.float16)\n"
                   "    return [x, (256,), torch.randn(256, device='cuda', dtype=torch.float16)]\n"),
         note="naive=T.serial sum-of-squares + fp32 I/O, opt=T.reduce + fp16 I/O + torch.empty"),
    dict(kernel="argmax", dsl="tilelang", round=1, bound="memory",
         shape="x:(8192,32768) dim=1 fp16", extra=(),
         smoke=_gi("    return [torch.randn(128, 512, device='cuda', dtype=torch.float16), 1]\n"),
         note="naive=3D T.serial element scan + fp32 upcast, opt=2D shared-mem tiled scan + fp16 I/O"),
    dict(kernel="matmul", dsl="triton", round=1, bound="compute",
         shape="a,b:(4096,4096) fp16", extra=(),
         smoke=_gi("    a = torch.randn(256, 256, device='cuda', dtype=torch.float16)\n"
                   "    return [a, torch.randn(256, 256, device='cuda', dtype=torch.float16)]\n"),
         note="naive=fixed 64x64x64 blocks, no autotune; opt=@autotune 12 cfgs + GROUP_SIZE_M L2 swizzle"),
    dict(kernel="conv2d", dsl="triton", round=1, bound="compute",
         shape="x:(32,256,128,128) w:(256,256,3,3) pad=1 fp16", extra=(None, 1, 1),
         smoke=_gi("    x = torch.randn(2, 32, 32, 32, device='cuda', dtype=torch.float16)\n"
                   "    return [x, torch.randn(32, 32, 3, 3, device='cuda', dtype=torch.float16)]\n"),
         note="padding=1 per benchmark.py:129. naive=direct h/w/c loops; opt=padded implicit GEMM + autotune + fp16 TC"),

    # ---- Round 2 (naive = ViperBench @ a72e84b; opt = current ViperBench) ----
    dict(kernel="max_reduction", dsl="tilelang", round=2, bound="memory",
         shape="x:(8192,32768) dim=1 fp16", extra=(),
         smoke=_gi("    return [torch.randn(128, 4096, device='cuda', dtype=torch.float16), 1]\n"),
         note="naive=3D T.serial scan + fp32 upcast, opt=two-pass tiled T.reduce (value,index) + native fp16"),
    dict(kernel="mean_reduction", dsl="tilelang", round=2, bound="memory",
         shape="x:(8192,32768) dim=1 fp32", extra=(),
         smoke=_gi("    return [torch.randn(128, 4096, device='cuda', dtype=torch.float32), 1]\n"),
         note="naive=3D T.serial accumulate, opt=tiled T.reduce one-block-per-row"),
    dict(kernel="softmax", dsl="tilelang", round=2, bound="memory",
         shape="x:(4096,32768) fp16", extra=(),
         smoke=_gi("    return [torch.randn(128, 4096, device='cuda', dtype=torch.float16)]\n"),
         note="CAVEAT: naive falls back to torch.softmax(fp32) for N>8192 (=> PyTorch call, not a TileLang kernel "
              "at this large shape); opt=shared-mem row cache + tiled T.reduce + native fp16"),
    dict(kernel="log_softmax", dsl="tilelang", round=2, bound="memory",
         shape="x:(4096,32768) fp16", extra=(),
         smoke=_gi("    return [torch.randn(128, 4096, device='cuda', dtype=torch.float16)]\n"),
         note="naive=whole-row fp32 fragment kernel (spills), opt=shared-mem row cache + tiled reduce + native fp16"),
    dict(kernel="max_reduction", dsl="triton", round=2, bound="memory",
         shape="x:(8192,32768) dim=1 fp16", extra=(),
         smoke=_gi("    return [torch.randn(128, 4096, device='cuda', dtype=torch.float16), 1]\n"),
         note="naive=single block loads whole next_pow2(N) row (spills), opt=tiled streaming reduction BLOCK_N=4096"),
    dict(kernel="batched_matmul", dsl="tilelang", round=2, bound="memory",
         shape="A:(128,2048) B:(128,2048,2048) fp16", extra=(),
         smoke=_gi("    A = torch.randn(8, 256, device='cuda', dtype=torch.float16)\n"
                   "    return [A, torch.randn(8, 64, 256, device='cuda', dtype=torch.float16)]\n"),
         note="GEMV-like (AI~1 FLOP/B). naive=scalar T.serial MAC + fp32 upcast, opt=shared-A cache + tiled T.reduce + native fp16"),
    dict(kernel="conv2d", dsl="tilelang", round=2, bound="compute",
         shape="x:(32,256,128,128) w:(256,256,3,3) pad=1 fp16", extra=(None, 1, 1),
         smoke=_gi("    x = torch.randn(2, 32, 32, 32, device='cuda', dtype=torch.float16)\n"
                   "    return [x, torch.randn(32, 32, 3, 3, device='cuda', dtype=torch.float16)]\n"),
         note="padding=1 per benchmark.py:129 (OW=128 triggers opt fast path). naive=im2col + per-batch GEMM loop, "
              "opt=direct implicit conv as KH*KW accumulating fp16 TC GEMMs"),
]


# ---------------------------------------------------------------------------
# Per-GPU roofline peaks.  DOCUMENTED ASSUMPTIONS (vendor dense specs, no
# sparsity; FP16/BF16 tensor cores w/ FP32 accumulate). Matched by substring of
# the runtime device name (device_slug). If no entry matches, roofline_frac is
# left blank with an explanatory note (we do NOT guess silently).
#   memory      = peak HBM bandwidth, bytes/s
#   compute_fp16= peak dense FP16/BF16 tensor-core throughput, FLOP/s
# ---------------------------------------------------------------------------
PEAKS = {
    "GH200": dict(
        memory=4.00e12, compute_fp16=989.5e12,
        note="GH200 480GB: 96GB HBM3 ~4.0 TB/s; Hopper dense FP16/BF16 TC (FP32 acc) ~989.5 TFLOP/s, no sparsity. "
             "(The 144GB HBM3e variant is ~4.9 TB/s; 4.0 TB/s used here.)"),
    "H100": dict(
        memory=3.35e12, compute_fp16=989.5e12,
        note="H100 80GB HBM3 ~3.35 TB/s; Hopper dense FP16 TC ~989.5 TFLOP/s, no sparsity."),
    "A100": dict(
        memory=1.555e12, compute_fp16=312e12,
        note="A100 40GB HBM2e ~1.555 TB/s (80GB SXM is ~2.039 TB/s); Ampere dense FP16 TC ~312 TFLOP/s, no sparsity."),
}


def select_peak(slug: str, bound_type: str):
    """Return (peak_value, assumed_str, gpu_note) or (None, reason, '') if no entry."""
    s = slug.upper()
    for key, tbl in PEAKS.items():
        if key in s:
            if bound_type == "memory":
                return tbl["memory"], f"{tbl['memory']:.4g} B/s HBM ({key})", tbl["note"]
            if bound_type == "compute":
                return tbl["compute_fp16"], f"{tbl['compute_fp16']:.4g} FLOP/s FP16-TC ({key})", tbl["note"]
            return None, f"unknown bound_type '{bound_type}'", tbl["note"]
    return (None,
            f"no peak-table entry for GPU '{slug}' (add it to PEAKS before trusting roofline_frac)",
            "")


# ---------------------------------------------------------------------------
# Essential (minimal) DRAM traffic / FLOPs for the op -- the roofline numerator.
# We deliberately count the MINIMAL traffic an ideal algorithm must move (read
# inputs once + write outputs once) / the exact math FLOPs, NOT each variant's
# actual passes. roofline_frac_opt = essential_work / (t_opt * peak) then reads
# as "fraction of the hardware roofline the optimized kernel reaches assuming the
# minimal-traffic algorithm" -- a conservative, baseline-independent quality bar.
# ---------------------------------------------------------------------------
def essential_work(kernel: str, inputs):
    el = lambda t: t.element_size()
    if kernel in ("layer_norm", "rms_norm", "softmax", "log_softmax"):
        x = inputs[0]
        b = 2 * x.numel() * el(x)
        return "memory", b, f"read x + write y = 2*{x.numel()}*{el(x)}B"
    if kernel in ("argmax", "max_reduction"):
        x = inputs[0]
        b = x.numel() * el(x)
        return "memory", b, f"read x once = {x.numel()}*{el(x)}B (output negligible)"
    if kernel == "mean_reduction":
        x = inputs[0]
        b = x.numel() * el(x)
        return "memory", b, f"read x once = {x.numel()}*{el(x)}B"
    if kernel == "batched_matmul":
        A, B = inputs[0], inputs[1]
        out = A.shape[0] * B.shape[1]
        b = (A.numel() + B.numel() + out) * el(A)
        return "memory", b, f"read A+B + write C = ({A.numel()}+{B.numel()}+{out})*{el(A)}B (B dominates)"
    if kernel == "matmul":
        a, bmat = inputs[0], inputs[1]
        M, K = a.shape
        N = bmat.shape[1]
        f = 2 * M * N * K
        return "compute", f, f"2*M*N*K = 2*{M}*{N}*{K}"
    if kernel == "conv2d":
        x, w = inputs[0], inputs[1]
        N, C, H, W = x.shape
        OC, _, KH, KW = w.shape
        pad, stride = 1, 1  # matches extra=(None, 1, 1)
        OH = (H + 2 * pad - KH) // stride + 1
        OW = (W + 2 * pad - KW) // stride + 1
        f = 2 * N * OC * OH * OW * C * KH * KW
        return "compute", f, f"2*N*OC*OH*OW*C*KH*KW = 2*{N}*{OC}*{OH}*{OW}*{C}*{KH}*{KW} (pad=1)"
    return "unknown", 0, "no work model"


# ---------------------------------------------------------------------------
# Variant loading + timing.
# ---------------------------------------------------------------------------
def load_variant(variant: str, kernel: str, dsl: str, fn_name: str):
    """Import experiments/cliff/<variant>/<kernel>_<dsl>.py and return its fn.
    ViperBench is on sys.path (via _harness) so the copied impls' guarded
    `from tuning.cache import ...` resolves; on a GPU with no cached config it
    transparently falls back to the naive defaults (identical to ViperBench)."""
    base = NAIVE_DIR if variant == "naive" else OPT_DIR
    path = base / f"{kernel}_{dsl}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    modname = f"cliff_{variant}_{kernel}_{dsl}"
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)


def build_inputs(get_inputs_src: str):
    ns = {"torch": torch}
    exec(get_inputs_src, ns)
    return ns["get_inputs"]()


def time_variant(fn, get_inputs_src, extra, warmup, reps):
    """Build fresh inputs (avoids cross-call aliasing/caching) and time fn."""
    inputs = build_inputs(get_inputs_src)
    t = time_kernel(fn, *inputs, *extra, warmup=warmup, reps=reps)
    return t["median_ms"]


def _ratio(num, den):
    if num is None or den is None or den != den or num != num or den <= 0:
        return ""
    return round(num / den, 4)


def run_entry(entry, smoke, warmup, reps, slug):
    kernel, dsl = entry["kernel"], entry["dsl"]
    config = KERNEL_CONFIGS[kernel]
    fn_name = config["fn"]
    dtype_label = PRECISION_TO_LABEL.get(config["precision"], config["precision"])
    extra = entry["extra"]
    get_inputs_src = entry["smoke"] if smoke else config["get_inputs"]
    shape = (entry["shape"] + " [SMOKE]") if smoke else entry["shape"]

    print(f"\n--- {kernel} ({dsl}, {dtype_label}) @ {shape} ---")

    notes = [entry["note"]]
    t_lib = t_naive = t_opt = float("nan")

    # Library (PyTorch / vendor) reference.
    try:
        lib_fn = getattr(load_impl(kernel, "pytorch"), fn_name)
        t_lib = time_variant(lib_fn, get_inputs_src, extra, warmup, reps)
        print(f"    library : {t_lib:.5f} ms")
    except Exception as e:
        notes.append(f"library_error: {type(e).__name__}: {str(e)[:90]}")
        print(f"    [warn] library {type(e).__name__}: {str(e)[:90]}")

    # Naive DSL.
    try:
        naive_fn = load_variant("naive", kernel, dsl, fn_name)
        t_naive = time_variant(naive_fn, get_inputs_src, extra, warmup, reps)
        print(f"    naive   : {t_naive:.5f} ms")
    except Exception as e:
        notes.append(f"naive_error: {type(e).__name__}: {str(e)[:90]}")
        print(f"    [warn] naive {type(e).__name__}: {str(e)[:90]}")

    # Optimized DSL.
    try:
        opt_fn = load_variant("opt", kernel, dsl, fn_name)
        t_opt = time_variant(opt_fn, get_inputs_src, extra, warmup, reps)
        print(f"    optimized: {t_opt:.5f} ms")
    except Exception as e:
        notes.append(f"opt_error: {type(e).__name__}: {str(e)[:90]}")
        print(f"    [warn] opt {type(e).__name__}: {str(e)[:90]}")

    valid = lambda v: isinstance(v, float) and v == v and v > 0
    e_lib_naive = _ratio(t_lib if valid(t_lib) else None, t_naive if valid(t_naive) else None)
    e_lib_opt = _ratio(t_lib if valid(t_lib) else None, t_opt if valid(t_opt) else None)
    cliff = _ratio(t_naive if valid(t_naive) else None, t_opt if valid(t_opt) else None)

    # ---- Roofline anchor on the OPTIMIZED kernel (baseline-independent). ----
    bound = peak_assumed = ""
    roof_frac = ""
    if valid(t_opt):
        try:
            ins = build_inputs(get_inputs_src)
            bound, work, work_detail = essential_work(kernel, ins)
            del ins
            torch.cuda.empty_cache()
            peak, peak_assumed, gpu_note = select_peak(slug, bound)
            if peak is not None and work > 0:
                achieved = work / (t_opt / 1000.0)  # B/s or FLOP/s
                roof_frac = round(achieved / peak, 4)
                unit = "B/s" if bound == "memory" else "FLOP/s"
                notes.append(f"roofline[{bound}]: {work_detail}; achieved={achieved:.3g}{unit}; "
                             f"peak={peak:.3g}{unit}; {gpu_note}")
            else:
                notes.append(f"roofline[{bound}]: {peak_assumed}; {work_detail}")
        except Exception as e:
            notes.append(f"roofline_error: {type(e).__name__}: {str(e)[:80]}")

    if cliff != "":
        print(f"    => E_lib_naive={e_lib_naive}  E_lib_opt={e_lib_opt}  "
              f"cliff={cliff}x  roofline_frac_opt={roof_frac}")

    fmt = lambda v: round(v, 5) if (isinstance(v, float) and v == v) else ""
    return dict(
        kernel=kernel, dsl=dsl, dtype=dtype_label, shape=shape,
        t_lib_ms=fmt(t_lib), t_naive_ms=fmt(t_naive), t_opt_ms=fmt(t_opt),
        E_lib_naive=e_lib_naive, E_lib_opt=e_lib_opt, cliff=cliff,
        roofline_bound_type=bound, roofline_peak_assumed=peak_assumed,
        roofline_frac_opt=roof_frac, note=" | ".join(notes),
    )


def _print_summary(rows):
    print("\n" + "=" * 110)
    print("  SUMMARY — naive vs optimized vs library; cliff = t_naive/t_opt; "
          "roofline_frac_opt = baseline-independent quality anchor")
    print("=" * 110)
    hdr = (f"  {'kernel':<15}{'dsl':<9}{'dtype':<6}{'t_lib':>10}{'t_naive':>11}"
           f"{'t_opt':>10}{'E_lib_nv':>10}{'E_lib_opt':>11}{'cliff':>9}{'roof_opt':>10}")
    print(hdr)
    print("  " + "-" * 108)
    for r in rows:
        g = lambda k: (f"{r[k]:.3f}" if isinstance(r[k], (int, float)) else "-")
        cliff_str = f"{r['cliff']:.2f}x" if isinstance(r["cliff"], (int, float)) else "-"
        print(f"  {r['kernel']:<15}{r['dsl']:<9}{r['dtype']:<6}"
              f"{g('t_lib_ms'):>10}{g('t_naive_ms'):>11}{g('t_opt_ms'):>10}"
              f"{g('E_lib_naive'):>10}{g('E_lib_opt'):>11}"
              f"{cliff_str:>9}"
              f"{g('roofline_frac_opt'):>10}")
    print("  " + "-" * 108)
    print("  E_lib_* are ratios t_lib/t_variant (>1 => DSL faster than library). "
          "roof_opt is fraction of HBM/FP16-TC roofline.")
    print("=" * 110)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny shapes + few reps for a fast plumbing check (NOT for the paper)")
    ap.add_argument("--kernels", type=str, default=None,
                    help="comma-separated subset; match a kernel name (both dsls) or '<kernel>_<dsl>'")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this experiment requires a GPU.")
        sys.exit(2)

    if args.smoke:
        warmup = args.warmup if args.warmup is not None else 3
        reps = args.reps if args.reps is not None else 5
        print("[SMOKE] tiny shapes, few reps — plumbing only, NOT for the paper.")
    else:
        warmup = args.warmup if args.warmup is not None else 15
        reps = args.reps if args.reps is not None else 50

    entries = CLIFF_SET
    if args.kernels:
        wanted = {k.strip() for k in args.kernels.split(",")}
        entries = [e for e in CLIFF_SET
                   if e["kernel"] in wanted or f"{e['kernel']}_{e['dsl']}" in wanted]
        if not entries:
            avail = sorted({e["kernel"] for e in CLIFF_SET}
                           | {f"{e['kernel']}_{e['dsl']}" for e in CLIFF_SET})
            print(f"No cliff entries match {sorted(wanted)}; available: {avail}")
            sys.exit(2)

    info = banner("Dual-variant cliff + roofline anchor  (RQ2 gap / RQ3 heuristics)")
    slug = device_slug()
    print(f"  cliff set: {len(entries)} (kernel,dsl) pairs  |  warmup/reps = {warmup}/{reps}")
    print(f"  provenance: experiments/cliff/PROVENANCE.md  |  results -> results/{slug}/{EXPERIMENT}.csv")

    rows = []
    for entry in entries:
        try:
            rows.append(run_entry(entry, args.smoke, warmup, reps, slug))
        except Exception as e:  # pragma: no cover - keep sweeping on a single failure
            kernel, dsl = entry["kernel"], entry["dsl"]
            print(f"  [error] {kernel} ({dsl}): {type(e).__name__}: {str(e)[:120]}")
            rows.append(dict(
                kernel=kernel, dsl=dsl,
                dtype=PRECISION_TO_LABEL.get(KERNEL_CONFIGS[kernel]["precision"], "?"),
                shape=entry["shape"], t_lib_ms="", t_naive_ms="", t_opt_ms="",
                E_lib_naive="", E_lib_opt="", cliff="",
                roofline_bound_type="", roofline_peak_assumed="", roofline_frac_opt="",
                note=f"ENTRY_ERROR: {type(e).__name__}: {str(e)[:120]}"))
        torch.cuda.empty_cache()

    _print_summary(rows)
    write_csv(EXPERIMENT, rows, [
        "kernel", "dsl", "dtype", "shape",
        "t_lib_ms", "t_naive_ms", "t_opt_ms",
        "E_lib_naive", "E_lib_opt", "cliff",
        "roofline_bound_type", "roofline_peak_assumed", "roofline_frac_opt", "note",
    ])


if __name__ == "__main__":
    main()
