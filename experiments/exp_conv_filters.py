#!/usr/bin/env python3
"""
Experiment: Conv filter-size coverage sweep  (ASE-2026 #4134 rebuttal)
======================================================================

WHAT THIS ANSWERS
-----------------
Reviewer R1-Q2 and R2-Q2 / weakness **W6**: the paper claims conv coverage of
1x1-7x7 + depthwise + strided convolutions, but the artifact only benchmarks the
single 3x3 stride-1 case. This script sweeps the *full* advertised filter family
on a fixed, realistic input shape and reports library-efficiency (E_lib =
t_library / t_DSL * 100) for both DSLs (Triton, TileLang) against the PyTorch
(cuDNN) baseline -- turning an asserted coverage claim into measured data.

It also feeds two further points:
  * **RC3** (register pressure at larger filters, R1-Q4): the Triton conv kernel
    loops `for h in range(kernel_height): for w in range(kernel_width)`
    (ViperBench/conv2d/triton_impl.py:54-55) so the fp32 `accum` register
    lifetime grows with k^2. We capture the *real* kernel's `n_regs` / `n_spills`
    per filter size via Triton 3.4.0's CompiledKernel handle (admin-free: no ncu,
    no profiling permission needed -- this is "Experiment 3 Path A").
  * **W13** (Ada register numbers): the captured n_regs are the measured Ada
    (sm_89) register counts the reviewers asked for.

PORTABILITY (mandatory)
-----------------------
The SAME script runs unchanged on RTX 4000 Ada (20 GB, sm_89, now) and on
A100 / H100 (80 GB, later). Nothing is hardcoded to a specific architecture:
  * Device properties come from `_harness.device_info()` (queried at runtime).
  * The input shape is a CLI parameter with an Ada-safe default that fits in
    ~20 GB; on 80 GB cards you can pick `--shape large` (or pass a custom shape)
    to scale up. The 7x7 im2col working set is large, so every config is wrapped
    in a CUDA-OOM guard: an out-of-memory config is *skipped with a message*
    rather than crashing the whole sweep.
  * Results are auto-tagged by GPU under experiments/results/<gpu_slug>/ by the
    shared harness, so Ada / A100 / H100 outputs never collide.

USAGE
-----
    CUDA_VISIBLE_DEVICES=0 python experiments/exp_conv_filters.py            # default sweep (small shape, fp16)
    CUDA_VISIBLE_DEVICES=0 python experiments/exp_conv_filters.py --shape large
    CUDA_VISIBLE_DEVICES=0 python experiments/exp_conv_filters.py --smoke    # tiny shapes, quick (build-time only)
    CUDA_VISIBLE_DEVICES=0 python experiments/exp_conv_filters.py --reps 100

During development run ONLY `--smoke` (tiny shapes, <60 s, <2 GB). The full sweep
is run later by the orchestrator under controlled idle-GPU conditions.
"""
import argparse
import gc
import sys
from pathlib import Path

import torch

# Import the shared portable harness (sits next to this file).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import (  # noqa: E402
    banner,
    device_info,
    library_efficiency,
    load_impl,
    load_optimized,
    time_kernel,
    write_csv,
)


# ---------------------------------------------------------------------------
# Shape presets.  All NCHW.  "same" padding = k//2 for stride-1 cases.
#   * large : the paper's Table-2 conv-large case (needs ~80 GB headroom for the
#             7x7 im2col -> intended for A100/H100; OOM-guarded on Ada).
#   * small : a comfortably Ada-safe realistic shape (fits well under 20 GB).
#   * smoke : tiny, for build-time correctness/plumbing checks only.
# Channels are kept multiples of 32 so the Triton/TileLang block tiles divide
# evenly (BLOCK_SIZE_IN_FEAT / BLOCK_SIZE_OUT_FEAT default 32; GEMM blocks 64/32).
# ---------------------------------------------------------------------------
SHAPES = {
    "large": (32, 256, 128, 128),   # N, C, H, W  (paper large case)
    "small": (8, 64, 56, 56),       # Ada-safe realistic case
    "smoke": (2, 32, 16, 16),       # tiny build-time check
}


def build_configs(N, C, H, W):
    """Return the list of conv configs to sweep on the chosen input shape.

    Each entry: dict(name, filter, stride, padding, groups, weight_shape, dtype-note).
    weight_shape is (OC, C/groups, KH, KW).  We keep OC == C so every config has
    a square channel GEMM (comparable register/latency behavior across k).
    """
    OC = C
    cfgs = []
    # 1x1, 3x3, 5x5, 7x7  (stride 1, "same" padding = k//2)
    for k in (1, 3, 5, 7):
        cfgs.append(dict(
            name=f"{k}x{k}_s1", filt=f"{k}x{k}", k=k, stride=1, padding=k // 2,
            groups=1, weight_shape=(OC, C, k, k),
        ))
    # depthwise: groups == channels, weight (C, 1, k, k).  Exercises the groups>1
    # path in all three backends (PyTorch grouped conv; Triton group_pid loop;
    # TileLang per-group GEMM loop).
    cfgs.append(dict(
        name="dw_3x3_s1", filt="3x3", k=3, stride=1, padding=1,
        groups=C, weight_shape=(C, 1, 3, 3),
    ))
    # strided: 3x3 stride 2 (Winograd-ineligible; halves spatial output).
    cfgs.append(dict(
        name="3x3_s2", filt="3x3", k=3, stride=2, padding=1,
        groups=1, weight_shape=(OC, C, 3, 3),
    ))
    return cfgs


# ---------------------------------------------------------------------------
# Memory estimate + OOM helpers (portability: guard the big 7x7 im2col).
# ---------------------------------------------------------------------------
def out_hw(H, W, k, stride, padding):
    OH = (H + 2 * padding - k) // stride + 1
    OW = (W + 2 * padding - k) // stride + 1
    return OH, OW


def est_im2col_bytes(N, C, H, W, cfg, bytes_per_elem=4):
    """Rough peak working-set estimate for the TileLang im2col path (the memory
    hog).  col tensor is (N, C*KH*KW, OH*OW) and TileLang upcasts to fp32 (4B),
    plus padded GEMM buffers.  Used only to print a heads-up; the real guard is
    the try/except around CUDA OOM."""
    k = cfg["k"]
    OH, OW = out_hw(H, W, k, cfg["stride"], cfg["padding"])
    col_elems = N * C * k * k * OH * OW
    return col_elems * bytes_per_elem


def free_mem_bytes(idx=0):
    free, _total = torch.cuda.mem_get_info(idx)
    return free


def is_oom(exc) -> bool:
    """True if the exception is a CUDA out-of-memory (covers torch's dedicated
    OutOfMemoryError and the generic RuntimeError 'out of memory' text)."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# RC3: capture the REAL Triton conv kernel's n_regs / n_spills.
#
# The unified-API wrapper ViperBench/conv2d/triton_impl.py:conv2d() launches
# `conv2d_forward_kernel[grid](...)` but discards the returned CompiledKernel.
# We monkey-patch JITFunction.__getitem__ on that specific kernel object so the
# launch returns through a thin wrapper that records the handle. This captures
# the production kernel (preferred over an instrumented copy) with zero edits to
# any ViperBench file and no profiling permissions.
# ---------------------------------------------------------------------------
class TritonRegCapture:
    """Context manager that records the last CompiledKernel produced by the real
    conv2d_forward_kernel during a `triton_mod.conv2d(...)` call."""

    def __init__(self, triton_mod):
        self.kfn = getattr(triton_mod, "conv2d_forward_kernel", None)
        self.handle = None
        self._orig_getitem = None
        self._cls = None

    def __enter__(self):
        if self.kfn is None:
            return self
        self._cls = type(self.kfn)
        self._orig_getitem = self._cls.__getitem__
        cap = self

        def patched_getitem(kself, grid):
            launcher = cap._orig_getitem(kself, grid)

            def wrapped(*a, **kw):
                h = launcher(*a, **kw)
                # only record handles from OUR kernel object, not other kernels
                if kself is cap.kfn:
                    cap.handle = h
                return h

            return wrapped

        self._cls.__getitem__ = patched_getitem
        return self

    def __exit__(self, *exc):
        if self._cls is not None and self._orig_getitem is not None:
            self._cls.__getitem__ = self._orig_getitem
        return False

    def stats(self):
        """Return (n_regs, n_spills, num_warps) or (None, None, None)."""
        h = self.handle
        if h is None:
            return None, None, None
        md = getattr(h, "metadata", None)
        return (
            getattr(h, "n_regs", None),
            getattr(h, "n_spills", None),
            getattr(md, "num_warps", None) if md is not None else None,
        )


# ---------------------------------------------------------------------------
# Per-config driver.
# ---------------------------------------------------------------------------
def make_inputs(N, C, H, W, cfg, dtype, device="cuda"):
    """Allocate input + weight (+bias=None). weight is (OC, C/groups, KH, KW)."""
    x = torch.randn(N, C, H, W, device=device, dtype=dtype)
    w = torch.randn(*cfg["weight_shape"], device=device, dtype=dtype)
    return x, w


def correctness(ref, out, atol, rtol):
    rf, of = ref.float(), out.float()
    if rf.shape != of.shape:
        return False, float("inf")
    err = (rf - of).abs().max().item()
    ok = torch.allclose(rf, of, atol=atol, rtol=rtol)
    return ok, err


def run_config(impls, N, C, H, W, cfg, dtype, atol, rtol, warmup, reps, idx=0):
    """Run one conv config across pytorch/triton/tilelang.

    Returns a list of CSV row dicts (one per impl). Each row carries timing,
    E_lib (for DSLs), correctness, n_regs/n_spills (Triton only), and a note.
    OOM or backend errors are recorded in `note` (median_ms stays empty) instead
    of crashing the sweep.
    """
    pt, tri, tl = impls["pytorch"], impls["triton"], impls["tilelang"]
    kw = dict(stride=cfg["stride"], padding=cfg["padding"], groups=cfg["groups"])
    shape_str = f"{N}x{C}x{H}x{W}"
    base = dict(filter=cfg["filt"], stride=cfg["stride"], groups=cfg["groups"],
                shape=shape_str)

    OH, OW = out_hw(H, W, cfg["k"], cfg["stride"], cfg["padding"])
    est_gb = est_im2col_bytes(N, C, H, W, cfg) / 1024**3
    free_gb = free_mem_bytes(idx) / 1024**3
    print(f"\n[{cfg['name']}] {shape_str} -> out {N}x{C}x{OH}x{OW}  "
          f"groups={cfg['groups']}  (est im2col ~{est_gb:.2f} GB, free ~{free_gb:.1f} GB)")

    rows = []

    # Allocate input + weight ONCE and share them across all three backends, so
    # the DSL outputs are compared against a PyTorch reference computed on the
    # SAME data (otherwise correctness is meaningless). They stay alive for the
    # whole config and are freed at the end. If even the inputs don't fit, skip
    # the entire config (all three backends) rather than crash.
    try:
        x, w = make_inputs(N, C, H, W, cfg, dtype)
    except Exception as e:  # noqa: BLE001
        note = "OOM-skip(inputs)" if is_oom(e) else f"input-alloc-failed: {type(e).__name__}"
        print(f"  ALL: SKIP ({note})")
        free_cuda()
        for impl_name in ("pytorch", "triton", "tilelang"):
            rows.append(dict(base, impl=impl_name, median_ms="", mean_ms="",
                             std_ms="", e_lib="", n_regs="", n_spills="",
                             correct="", note=note))
        return rows

    # --- PyTorch (cuDNN) reference + timing -------------------------------
    ref = None
    t_pt = None
    pt_row = dict(base, impl="pytorch", median_ms="", mean_ms="", std_ms="",
                  e_lib="", n_regs="", n_spills="", correct="", note="")
    try:
        ref = pt.conv2d(x, w, **kw)
        torch.cuda.synchronize()
        t = time_kernel(pt.conv2d, x, w, warmup=warmup, reps=reps, **kw)
        t_pt = t["median_ms"]
        pt_row.update(median_ms=t["median_ms"], mean_ms=t["mean_ms"],
                      std_ms=t["std_ms"], e_lib=100.0, correct="ref")
        print(f"  pytorch (cuDNN): median {t['median_ms']:.4f} ms  "
              f"(mean {t['mean_ms']:.4f} +/- {t['std_ms']:.4f})")
    except Exception as e:  # noqa: BLE001
        note = "OOM-skip" if is_oom(e) else f"{type(e).__name__}: {str(e)[:80]}"
        pt_row.update(note=note)
        print(f"  pytorch (cuDNN): SKIP ({note})")
        free_cuda()
    rows.append(pt_row)

    # --- DSL backends (reuse the SAME x, w as PyTorch) --------------------
    # "mitigation" = the AKO4ALL optimized conv kernel; included only with
    # --mitigation, to test whether the RQ3 conv recovery generalizes (R1:68).
    dsl_backends = [("triton", tri), ("tilelang", tl)]
    mit = impls.get("mitigation")
    if mit is not None:
        dsl_backends.append(("mitigation", mit))
    for impl_name, mod in dsl_backends:
        row = dict(base, impl=impl_name, median_ms="", mean_ms="", std_ms="",
                   e_lib="", n_regs="", n_spills="", correct="", note="")
        # depthwise on TileLang launches N*groups serial GEMMs (im2col path);
        # flag it so a slow / costly number is interpretable, not surprising.
        notes = []
        if impl_name == "tilelang" and cfg["groups"] > 1:
            notes.append(f"per-group GEMM x{N*cfg['groups']} launches")
        # the optimized conv kernel asserts groups==1; skip depthwise cleanly
        # (this exclusion is itself the honest finding for R1:68).
        if impl_name == "mitigation" and cfg["groups"] > 1:
            row["note"] = "optimized kernel: groups==1 only (depthwise excluded)"
            print(f"  {impl_name:8s}: SKIP (depthwise unsupported, groups==1 only)")
            rows.append(row)
            continue
        try:
            # correctness + (for Triton) register capture in one warm call
            if impl_name == "triton":
                with TritonRegCapture(mod) as cap:
                    out = mod.conv2d(x, w, **kw)
                    torch.cuda.synchronize()
                n_regs, n_spills, num_warps = cap.stats()
                if n_regs is not None:
                    row.update(n_regs=n_regs, n_spills=n_spills)
                    notes.append(f"warps={num_warps}")
                    print(f"  triton  RC3: n_regs={n_regs} n_spills={n_spills} "
                          f"num_warps={num_warps}")
                else:
                    notes.append("reg-capture-miss")
            else:
                out = mod.conv2d(x, w, **kw)
                torch.cuda.synchronize()

            ok = err = None
            if ref is not None:
                ok, err = correctness(ref, out, atol, rtol)
                row.update(correct=bool(ok))
                if not ok:
                    # Note when only an fp32 input would match: the DSLs use
                    # fp16 inputs with long C*k*k accumulation chains, so an
                    # fp16 mismatch here is a precision artifact, not a wrong op.
                    notes.append(f"fp16-mismatch(maxerr={err:.2e}); rerun --dtype float32 to confirm op-correct")
            del out

            t = time_kernel(mod.conv2d, x, w, warmup=warmup, reps=reps, **kw)
            row.update(median_ms=t["median_ms"], mean_ms=t["mean_ms"],
                       std_ms=t["std_ms"])
            if t_pt is not None:
                row.update(e_lib=library_efficiency(t_pt, t["median_ms"]))
            errtxt = "" if err is None else f" maxerr={err:.2e}"
            print(f"  {impl_name:8s}: median {t['median_ms']:.4f} ms  "
                  f"(mean {t['mean_ms']:.4f} +/- {t['std_ms']:.4f})  "
                  f"E_lib={row['e_lib']}%  correct={row['correct']}{errtxt}")
        except Exception as e:  # noqa: BLE001
            note = "OOM-skip" if is_oom(e) else f"unsupported/failed: {type(e).__name__}: {str(e)[:70]}"
            notes.append(note)
            print(f"  {impl_name:8s}: SKIP ({note})")
            free_cuda()
        row["note"] = "; ".join(notes)
        rows.append(row)

    del x, w

    free_cuda()
    return rows


# ---------------------------------------------------------------------------
# Summary table.
# ---------------------------------------------------------------------------
def print_summary(rows):
    print("\n" + "=" * 96)
    print("  CONV FILTER SWEEP SUMMARY  (E_lib = t_pytorch / t_DSL * 100;  >100% = DSL faster)")
    print("=" * 96)
    hdr = (f"  {'filter':7s} {'s':>1s} {'grp':>5s} | "
           f"{'pytorch ms':>11s} | {'triton ms':>10s} {'E%':>7s} {'regs':>5s} {'spill':>5s} | "
           f"{'tilelang ms':>12s} {'E%':>7s} | corr")
    print(hdr)
    print("  " + "-" * 92)
    # group rows by (filter, stride, groups)
    by_cfg = {}
    order = []
    for r in rows:
        key = (r["filter"], r["stride"], r["groups"], r["shape"])
        if key not in by_cfg:
            by_cfg[key] = {}
            order.append(key)
        by_cfg[key][r["impl"]] = r

    def ms(r):
        v = r.get("median_ms", "")
        return f"{v:.4f}" if isinstance(v, (int, float)) else "SKIP"

    for key in order:
        grp = by_cfg[key]
        pt = grp.get("pytorch", {})
        tr = grp.get("triton", {})
        tlr = grp.get("tilelang", {})
        filt, stride, groups, _shape = key
        corr_bits = []
        for nm, r in (("T", tr), ("L", tlr)):
            c = r.get("correct", "")
            corr_bits.append(f"{nm}={'ok' if c is True else ('x' if c is False else '-')}")
        print(f"  {filt:7s} {stride:>1d} {groups:>5d} | "
              f"{ms(pt):>11s} | "
              f"{ms(tr):>10s} {str(tr.get('e_lib','')):>7s} "
              f"{str(tr.get('n_regs','')):>5s} {str(tr.get('n_spills','')):>5s} | "
              f"{ms(tlr):>12s} {str(tlr.get('e_lib','')):>7s} | "
              f"{' '.join(corr_bits)}")
    print("=" * 96)

    # RC3 focus line: registers vs filter size (stride-1, groups==1 dense convs)
    print("\n  RC3 (register pressure vs filter size, Triton dense stride-1 conv):")
    dense = [by_cfg[k].get("triton", {}) for k in order
             if k[1] == 1 and k[2] == 1]
    for r in dense:
        nr, ns = r.get("n_regs", ""), r.get("n_spills", "")
        spill = "" if ns == "" else (f"  SPILLS={ns}" if (isinstance(ns, int) and ns > 0) else " no-spill")
        print(f"    {r.get('filter','?'):5s}: n_regs={nr}  n_spills={ns}{spill}")
    print("    (n_regs is expected to rise with k^2; first n_spills>0 substantiates RC3 on this GPU)")


def print_mitigation_summary(rows):
    """R1:68 -- does the RQ3 conv mitigation generalize across filter sizes?
    Per config, show baseline-Triton E_lib next to the optimized-kernel E_lib
    (E_lib = t_cuDNN / t_kernel * 100; higher = closer to the cuDNN baseline)."""
    print("\n" + "=" * 84)
    print("  CONV MITIGATION GENERALITY  (R1:68;  E_lib = t_cuDNN / t_kernel * 100, higher=better)")
    print("=" * 84)
    print(f"  {'filter':7s} {'s':>1s} {'grp':>4s} | {'baseline triton E%':>19s} | "
          f"{'optimized (mitig) E%':>21s}")
    print("  " + "-" * 80)
    by_cfg, order = {}, []
    for r in rows:
        key = (r["filter"], r["stride"], r["groups"])
        if key not in by_cfg:
            by_cfg[key] = {}
            order.append(key)
        by_cfg[key][r["impl"]] = r
    for key in order:
        grp = by_cfg[key]
        filt, stride, groups = key
        be = grp.get("triton", {}).get("e_lib", "")
        mrow = grp.get("mitigation", {})
        me, mnote = mrow.get("e_lib", ""), mrow.get("note", "")
        extra = f"   ({mnote})" if (me == "" and mnote) else ""
        print(f"  {filt:7s} {stride:>1d} {groups:>4d} | {str(be):>19s} | "
              f"{str(me):>21s}{extra}")
    print("=" * 84)
    print("  Generality holds if the optimized E% stays high as filter size grows;")
    print("  a fall-off localizes where the RQ3 recovery does/doesn't transfer.")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Conv filter-size coverage sweep (W6/RC3).")
    ap.add_argument("--shape", choices=["large", "small"], default="small",
                    help="input shape preset (default: small, Ada-safe). "
                         "large = paper Table-2 case (needs ~80 GB; OOM-guarded on Ada).")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny shapes, fp16, quick: build-time plumbing/correctness check only.")
    ap.add_argument("--reps", type=int, default=None,
                    help="timing reps (default 50; smoke uses 5).")
    ap.add_argument("--warmup", type=int, default=None,
                    help="timing warmup iters (default 15; smoke uses 3).")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                    help="input dtype (default float16; smoke forces float16).")
    ap.add_argument("--mitigation", action="store_true",
                    help="also time the AKO4ALL optimized conv kernel "
                         "(results/optimized/conv2d_triton.py) to test whether the RQ3 "
                         "conv mitigation generalizes across filter sizes (R1:68). "
                         "Optimized kernel is groups==1 only (depthwise excluded); "
                         "writes conv_mitigation.csv.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    smoke = args.smoke
    shape_key = "smoke" if smoke else args.shape
    N, C, H, W = SHAPES[shape_key]
    dtype = torch.float16 if (smoke or args.dtype == "float16") else torch.float32
    reps = args.reps if args.reps is not None else (5 if smoke else 50)
    warmup = args.warmup if args.warmup is not None else (3 if smoke else 15)

    # fp16 tolerance per CLAUDE.md table (atol 1e-3) loosened for conv reductions
    # (long fp16 accumulation chains over C*k*k); spec suggests atol=5e-2 for fp16.
    if dtype == torch.float16:
        atol, rtol = 5e-2, 5e-2
    else:
        atol, rtol = 1e-4, 1e-4

    info = banner("Conv filter-size coverage sweep (W6 / RC3 / R1-Q2 / R2-Q2)")
    print(f"  Shape preset: {shape_key} = (N={N}, C={C}, H={H}, W={W})  "
          f"dtype={str(dtype).split('.')[-1]}")
    print(f"  Timing: warmup={warmup}, reps={reps}   "
          f"Correctness tol: atol={atol}, rtol={rtol}")
    if shape_key == "small":
        print("  Note: 'small' is Ada-safe (<20 GB). On A100/H100 (80 GB) use "
              "--shape large for the paper Table-2 case.")
    print()

    impls = {name: load_impl("conv2d", name)
             for name in ("pytorch", "triton", "tilelang")}
    experiment = "conv_filters"
    if args.mitigation:
        impls["mitigation"] = load_optimized("conv2d_triton")
        experiment = "conv_mitigation"
        print("  + mitigation arm: AKO4ALL optimized conv2d_triton -- tests whether the")
        print("    RQ3 conv recovery generalizes across filters (R1:68); groups==1 only.\n")
    # Auto-suffix by shape so back-to-back small/large runs don't clobber each
    # other (paper REVISION notes cite conv_{filters,mitigation}_{small,large}.csv).
    if shape_key in ("small", "large"):
        experiment = f"{experiment}_{shape_key}"

    cfgs = build_configs(N, C, H, W)
    all_rows = []
    for cfg in cfgs:
        all_rows.extend(run_config(impls, N, C, H, W, cfg, dtype,
                                   atol, rtol, warmup, reps, idx=0))

    print_summary(all_rows)
    if args.mitigation:
        print_mitigation_summary(all_rows)

    fieldnames = ["filter", "stride", "groups", "shape", "impl",
                  "median_ms", "mean_ms", "std_ms", "e_lib",
                  "n_regs", "n_spills", "correct", "note"]
    write_csv(experiment, all_rows, fieldnames)

    # exit non-zero only if every backend failed every config (real breakage),
    # not for individual OOM/unsupported skips (those are expected & recorded).
    ran = sum(1 for r in all_rows
              if isinstance(r.get("median_ms"), (int, float)))
    if ran == 0:
        print("\nERROR: no configuration produced a timing (all skipped/failed).")
        sys.exit(1)
    print(f"\nDone: {ran} (config x impl) timings recorded across {len(cfgs)} configs.")


if __name__ == "__main__":
    main()
