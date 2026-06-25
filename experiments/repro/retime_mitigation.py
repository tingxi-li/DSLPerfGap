#!/usr/bin/env python3
"""retime_mitigation.py -- re-time ALL kernel mitigations on THIS GPU (locked clocks).

Covers the AKO4ALL campaign kernels (layer_norm, rms_norm, argmax, matmul) AND the
optimized reduction/softmax family (experiments/opt_kernels/: mean_reduction,
max_reduction, softmax, log_softmax, logsumexp). The unoptimized "naive" baselines
are READ from this arch's profile.<slug>.csv (run regen_profile.sh first), so the
script is GPU-agnostic. Writes experiments/results/<slug>/mitigation_retime.csv.

GOTCHA baked in: TileLang JIT needs the venv's bundled nvidia libs on LD_LIBRARY_PATH
(libnvrtc.so.12 / libcudart.so.12), set below before importing tilelang.

Run with:  /home/ubuntu/dslperf-venv/bin/python experiments/repro/retime_mitigation.py
"""
import os, sys, csv, time, importlib.util, math
_NV = "/home/ubuntu/dslperf-venv/lib/python3.10/site-packages/nvidia"
os.environ["LD_LIBRARY_PATH"] = ":".join(
    [f"{_NV}/{p}/lib" for p in ("cuda_nvrtc", "cuda_runtime", "cublas", "cudnn")]
    + [os.environ.get("LD_LIBRARY_PATH", "")])
os.environ["_LSE_REEXEC"] = "1"; os.environ["_SOFTMAX_REEXEC"] = "1"  # skip opt-kernel re-exec shims
import torch, torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import re
_NAME = torch.cuda.get_device_name(0)
SLUG = re.sub(r"[^A-Za-z0-9._-]+", "_", _NAME).strip("_")          # results dir (NVIDIA_...): _harness convention
PROF_NAME = re.sub(r"\s+", "-", _NAME.replace("NVIDIA ", "")).strip("-")  # ViperBench profile: short name (A100-SXM4-40GB, H100-80GB-HBM3)
PROFILE = f"{REPO}/ViperBench/results/profile.{PROF_NAME}.csv"
AKO = f"{REPO}/AKO4ALL/results/optimized"
OPT = f"{REPO}/experiments/opt_kernels"
OUT = f"{REPO}/experiments/results/{SLUG}/mitigation_retime.csv"


def med_ms(fn, warmup=10, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort(); return ts[len(ts) // 2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def profile_lat(kernel, size, impl):
    if not os.path.exists(PROFILE): return None
    for r in csv.DictReader(open(PROFILE)):
        if r["kernel"] == kernel and r["size"] == size and r["impl"] == impl:
            return float(r["latency_ms"])
    return None


rows = []
def record(name, naive_ms, pytorch_ms, opt_ms, err, correct, note):
    oe = (pytorch_ms / opt_ms * 100) if opt_ms else None
    ne = (pytorch_ms / naive_ms * 100) if naive_ms else None
    rec = (pytorch_ms / opt_ms / (pytorch_ms / naive_ms)) if (naive_ms and opt_ms) else None
    rows.append([name, f"{naive_ms:.4f}" if naive_ms else "", f"{pytorch_ms:.4f}", f"{opt_ms:.4f}",
                 f"{ne:.1f}" if ne else "", f"{oe:.1f}" if oe else "",
                 f"{naive_ms/opt_ms:.1f}" if naive_ms else "", f"{err:.2e}", str(correct), note])
    print(f"{name:16} naive={naive_ms or float('nan'):>9.3f} opt={opt_ms:>8.4f} "
          f"E_lib={oe:>7.1f}% (naive {ne or float('nan'):.1f}%)  err={err:.1e} {correct}")

print(f"# arch={SLUG}  profile={os.path.relpath(PROFILE, REPO)}")
print(f"{'kernel':16}{'':>11}{'':>14}")

# ---- AKO4ALL normalization / reduction (tilelang) ----
# pytorch_ms uses the canonical profile.<arch> baseline (RQ1-consistent), live fallback if absent.
m = load(f"{AKO}/layer_norm_tilelang.py", "ln"); x, w, b = m.get_inputs()
_lo = m.layer_norm(x, w, b).float(); _lr = F.layer_norm(x, (x.shape[-1],), w, b).float()
err = (_lo - _lr).abs().max().item()
record("layer_norm", profile_lat("layer_norm", "large", "tilelang"),
       profile_lat("layer_norm", "large", "pytorch") or med_ms(lambda: F.layer_norm(x, (x.shape[-1],), w, b)),
       med_ms(lambda: m.layer_norm(x, w, b)), err, torch.allclose(_lo, _lr, atol=2e-2, rtol=5e-2),  # bf16+random-bias slack
       "AKO4ALL T.reduce+native bf16")

m = load(f"{AKO}/rms_norm_tilelang.py", "rn"); x, ns, w = m.get_inputs()
def rms_ref(x, w): xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * w.float()).to(x.dtype)
err = (m.rms_norm(x, ns, w).float() - rms_ref(x, w).float()).abs().max().item()
record("rms_norm", profile_lat("rms_norm", "large", "tilelang"),
       profile_lat("rms_norm", "large", "pytorch") or med_ms(lambda: rms_ref(x, w)),
       med_ms(lambda: m.rms_norm(x, ns, w)), err, err < 2e-2, "AKO4ALL T.reduce+native fp16")

m = load(f"{AKO}/argmax_tilelang.py", "am"); x, dim = m.get_inputs()
ridx = torch.argmax(x, dim=1); match = (m.argmax(x, dim).view(-1) == ridx.view(-1)).float().mean().item()
record("argmax", profile_lat("argmax", "large", "tilelang"),
       profile_lat("argmax", "large", "pytorch") or med_ms(lambda: torch.argmax(x, dim=1)),
       med_ms(lambda: m.argmax(x, dim)), 1 - match, match > 0.999, "AKO4ALL tiled (returns indices)")

# ---- AKO4ALL matmul (triton, 4096^2); naive baseline = triton_plain from autotune_matmul.csv ----
m = load(f"{AKO}/matmul_triton.py", "mm"); a, bb = m.get_inputs()
naive_mm = cublas_mm = None
amp = f"{REPO}/experiments/results/{SLUG}/autotune_matmul.csv"
if os.path.exists(amp):
    for r in csv.DictReader(open(amp)):
        if r.get("shape") == "4096x4096":
            if r.get("impl") == "triton_plain": naive_mm = float(r["median_ms"])
            if r.get("impl") == "cublas": cublas_mm = float(r["median_ms"])
err = (m.matmul(a, bb).float() - (a @ bb).float()).abs().max().item()
record("matmul", naive_mm, cublas_mm or med_ms(lambda: torch.matmul(a, bb)), med_ms(lambda: m.matmul(a, bb)),
       err, err < 0.2, "AKO4ALL autotune+L2 swizzle @4096^2")

# ---- optimized reduction / softmax family (experiments/opt_kernels/) ----
RED = [  # name, pytorch-ref, correctness atol
    ("mean_reduction", lambda x: x.mean(dim=1), 2e-3),
    ("max_reduction",  lambda x: torch.max(x, dim=1).values, 2e-2),
    ("logsumexp",      lambda x: torch.logsumexp(x, dim=-1), 2e-3),
    ("softmax",        lambda x: torch.softmax(x, dim=-1), 2e-2),     # fp16 baseline (paper-consistent)
    ("log_softmax",    lambda x: torch.log_softmax(x, dim=-1), 2e-2),
]
for name, ref, atol in RED:
    m = load(f"{OPT}/{name}_opt.py", f"{name}_opt"); x = m.get_inputs()[0]
    out = m.run(x); rf = ref(x)
    # correctness vs a high-precision (fp32) reference for the softmax variants
    hi = ref(x.float()).to(x.dtype) if x.dtype == torch.float16 and "softmax" in name else rf
    err = (out.float() - hi.float()).abs().max().item()
    record(name, profile_lat(name, "large", "tilelang"),
           profile_lat(name, "large", "pytorch") or med_ms(lambda: ref(x)),
           med_ms(lambda: m.run(x)), err, torch.allclose(out.float(), hi.float(), atol=atol, rtol=atol),
           "opt_kernels T.reduce + tiled streaming")

with open(OUT, "w", newline="") as f:
    w_ = csv.writer(f)
    w_.writerow(["kernel", "naive_ms", "pytorch_ms", "optimized_ms", "naive_elib_pct",
                 "optimized_elib_pct", "recovery_x", "max_abs_err", "correct", "note"])
    w_.writerows(rows)
print("\nWROTE", os.path.relpath(OUT, REPO))
