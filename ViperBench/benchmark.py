#!/usr/bin/env python3
"""
Profile latency and peak GPU memory for all ViperBench kernels.
Uses input shapes from kernel_input_shapes.html.
Outputs results to ViperBench/results/profile.csv
"""
import csv
import importlib
import sys
import time
import traceback
from pathlib import Path

import torch

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_ITERS = 10
MEASURE_ITERS = 100
# For large inputs where PyTorch ref is a pure-python loop, limit iterations
MEASURE_ITERS_SLOW = 5


def profile_fn(fn, args, kwargs=None, warmup=WARMUP_ITERS, iters=MEASURE_ITERS):
    """Profile a function's latency and peak GPU memory.
    Returns (median_latency_ms, peak_memory_mb).
    """
    kwargs = kwargs or {}

    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Measure latency
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median_ms = times[len(times) // 2]

    # Measure peak memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)

    return median_ms, peak_mem_mb


def load_module(kernel_dir, module_name):
    """Import a module from a kernel directory."""
    mod_path = kernel_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Per-kernel test case definitions ─────────────────────────────────────────
# Each entry: list of (size_label, fn_name, args, kwargs, description, is_slow)

M64 = 64 * 1024 * 1024  # 64M elements


def get_test_cases():
    """Build all test cases from kernel_input_shapes.html specifications."""
    cases = {}

    # add: x + y elementwise
    cases["add"] = [
        ("small", "add",
         (torch.randn(4096, device="cuda", dtype=torch.float16),
          torch.randn(4096, device="cuda", dtype=torch.float16)),
         None, "x:(4096,) y:(4096,) fp16", False),
        ("large", "add",
         (torch.randn(M64, device="cuda", dtype=torch.float16),
          torch.randn(M64, device="cuda", dtype=torch.float16)),
         None, "x:(64M,) y:(64M,) fp16", False),
    ]

    # argmax: reduce dim=1
    cases["argmax"] = [
        ("small", "argmax",
         (torch.randn(1024, 1024, device="cuda", dtype=torch.float16), 1),
         None, "x:(1024,1024) dim=1 fp16", False),
        ("large", "argmax",
         (torch.randn(8192, 32768, device="cuda", dtype=torch.float16), 1),
         None, "x:(8192,32768) dim=1 fp16", False),
    ]

    # attention: chunked linear attention, Q/K/V = (B,H,T,D)
    # Note: our attention impl uses float32 internally; large case reduced to fit GPU mem
    cases["attention"] = [
        ("small", "attention_fwd",
         (torch.randn(8, 16, 512, 64, device="cuda", dtype=torch.float32),
          torch.randn(8, 16, 512, 64, device="cuda", dtype=torch.float32),
          torch.randn(8, 16, 512, 64, device="cuda", dtype=torch.float32)),
         None, "QKV:(8,16,512,64) fp32", True),
        ("large", "attention_fwd",
         (torch.randn(8, 32, 2048, 128, device="cuda", dtype=torch.float32),
          torch.randn(8, 32, 2048, 128, device="cuda", dtype=torch.float32),
          torch.randn(8, 32, 2048, 128, device="cuda", dtype=torch.float32)),
         None, "QKV:(8,32,2048,128) fp32", True),
    ]

    # batched_matmul: our API is batched_matmul(A:[M,K], B:[M,N,K]) -> [M,N]
    cases["batched_matmul"] = [
        ("small", "batched_matmul",
         (torch.randn(64, 128, device="cuda", dtype=torch.float16),
          torch.randn(64, 128, 128, device="cuda", dtype=torch.float16)),
         None, "A:(64,128) B:(64,128,128) fp16", False),
        ("large", "batched_matmul",
         (torch.randn(128, 2048, device="cuda", dtype=torch.float16),
          torch.randn(128, 2048, 2048, device="cuda", dtype=torch.float16)),
         None, "A:(128,2048) B:(128,2048,2048) fp16", False),
    ]

    # conv2d: x=(N,C,H,W) w=(Cout,Cin,kH,kW), padding=1
    cases["conv2d"] = [
        ("small", "conv2d",
         (torch.randn(8, 64, 56, 56, device="cuda", dtype=torch.float16),
          torch.randn(64, 64, 3, 3, device="cuda", dtype=torch.float16)),
         {"padding": 1}, "x:(8,64,56,56) w:(64,64,3,3) fp16", False),
        ("large", "conv2d",
         (torch.randn(32, 256, 128, 128, device="cuda", dtype=torch.float16),
          torch.randn(256, 256, 3, 3, device="cuda", dtype=torch.float16)),
         {"padding": 1}, "x:(32,256,128,128) w:(256,256,3,3) fp16", False),
    ]

    # cross_entropy: our API is cross_entropy_fwd(logits, labels, smoothing, logit_scale, ...)
    # PyTorch ref is pure python loop - very slow on large
    cases["cross_entropy"] = [
        ("small", "cross_entropy_fwd",
         (torch.randn(256, 1024, device="cuda", dtype=torch.float32),
          torch.randint(0, 1024, (256,), device="cuda", dtype=torch.int64),
          0.0, 1.0, 0.0, -100, 1024, 0, 256, False, False),
         None, "logits:(256,1024) fp32", True),
        ("large", "cross_entropy_fwd",
         (torch.randn(4096, 32768, device="cuda", dtype=torch.float32),
          torch.randint(0, 32768, (4096,), device="cuda", dtype=torch.int64),
          0.0, 1.0, 0.0, -100, 32768, 0, 1024, False, False),
         None, "logits:(4096,32768) fp32", True),
    ]

    # embedding: embedding(input_ids, weight, vob_start, vob_end, out)
    cases["embedding"] = [
        ("small", "embedding",
         (torch.randint(0, 8192, (2048,), device="cuda", dtype=torch.int32),
          torch.randn(8192, 256, device="cuda", dtype=torch.float16),
          0, 8192,
          torch.zeros(2048, 256, device="cuda", dtype=torch.float16)),
         None, "w:(8192,256) idx:(2048,) fp16", False),
        ("large", "embedding",
         (torch.randint(0, 131072, (131072,), device="cuda", dtype=torch.int32),
          torch.randn(131072, 1024, device="cuda", dtype=torch.float16),
          0, 131072,
          torch.zeros(131072, 1024, device="cuda", dtype=torch.float16)),
         None, "w:(131072,1024) idx:(131072,) fp16", False),
    ]

    # index_select: index_select(output, source, index)
    cases["index_select"] = [
        ("small", "index_select",
         (torch.empty(256, 512, device="cuda", dtype=torch.float16),
          torch.randn(4096, 512, device="cuda", dtype=torch.float16),
          torch.randint(0, 4096, (256,), device="cuda", dtype=torch.int64)),
         None, "src:(4096,512) idx:(256,) fp16", False),
        ("large", "index_select",
         (torch.empty(4096, 2048, device="cuda", dtype=torch.float16),
          torch.randn(65536, 2048, device="cuda", dtype=torch.float16),
          torch.randint(0, 65536, (4096,), device="cuda", dtype=torch.int64)),
         None, "src:(65536,2048) idx:(4096,) fp16", False),
    ]

    # layer_norm: layer_norm(x, weight, bias) - uses bfloat16 (kernel constraint)
    cases["layer_norm"] = [
        ("small", "layer_norm",
         (torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16),
          torch.randn(1024, device="cuda", dtype=torch.bfloat16),
          torch.randn(1024, device="cuda", dtype=torch.bfloat16)),
         None, "x:(512,1024) bf16", False),
        ("large", "layer_norm",
         (torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16),
          torch.randn(8192, device="cuda", dtype=torch.bfloat16),
          torch.randn(8192, device="cuda", dtype=torch.bfloat16)),
         None, "x:(8192,8192) bf16", False),
    ]

    # leaky_relu: leaky_relu(a, b, activation) = matmul + activation
    cases["leaky_relu"] = [
        ("small", "leaky_relu",
         (torch.randn(4096, 4096, device="cuda", dtype=torch.float16),
          torch.randn(4096, 4096, device="cuda", dtype=torch.float16),
          "leaky_relu"),
         None, "a:(4096,4096) b:(4096,4096) fp16", False),
        ("large", "leaky_relu",
         (torch.randn(8192, 8192, device="cuda", dtype=torch.float16),
          torch.randn(8192, 8192, device="cuda", dtype=torch.float16),
          "leaky_relu"),
         None, "a:(8192,8192) b:(8192,8192) fp16", False),
    ]

    # linear_activation: kernel_ff(x, w1, w3, rms_w) - Llama FFN
    # x must be 3D: (batch, seq, dim)
    cases["linear_activation"] = [
        ("small", "kernel_ff",
         (torch.randn(1, 256, 1024, device="cuda", dtype=torch.float16),
          torch.randn(4096, 1024, device="cuda", dtype=torch.float16),
          torch.randn(4096, 1024, device="cuda", dtype=torch.float16),
          torch.randn(1024, device="cuda", dtype=torch.float16)),
         None, "x:(1,256,1024) w1/w3:(4096,1024) fp16", True),
        ("large", "kernel_ff",
         (torch.randn(1, 2048, 4096, device="cuda", dtype=torch.float16),
          torch.randn(16384, 4096, device="cuda", dtype=torch.float16),
          torch.randn(16384, 4096, device="cuda", dtype=torch.float16),
          torch.randn(4096, device="cuda", dtype=torch.float16)),
         None, "x:(1,2048,4096) w1/w3:(16384,4096) fp16", True),
    ]

    # log_softmax: log_softmax(x)
    cases["log_softmax"] = [
        ("small", "log_softmax",
         (torch.randn(512, 1024, device="cuda", dtype=torch.float16),),
         None, "x:(512,1024) fp16", False),
        ("large", "log_softmax",
         (torch.randn(4096, 32768, device="cuda", dtype=torch.float16),),
         None, "x:(4096,32768) fp16", False),
    ]

    # logsumexp: logsumexp(x) - fp32 for numerical stability
    cases["logsumexp"] = [
        ("small", "logsumexp",
         (torch.randn(512, 1024, device="cuda", dtype=torch.float32),),
         None, "x:(512,1024) fp32", False),
        ("large", "logsumexp",
         (torch.randn(8192, 16384, device="cuda", dtype=torch.float32),),
         None, "x:(8192,16384) fp32", False),
    ]

    # matmul: matmul(a, b) - 2D fp16
    cases["matmul"] = [
        ("small", "matmul",
         (torch.randn(512, 512, device="cuda", dtype=torch.float16),
          torch.randn(512, 512, device="cuda", dtype=torch.float16)),
         None, "A:(512,512) B:(512,512) fp16", False),
        ("large", "matmul",
         (torch.randn(16384, 16384, device="cuda", dtype=torch.float16),
          torch.randn(16384, 16384, device="cuda", dtype=torch.float16)),
         None, "A:(16384,16384) B:(16384,16384) fp16", False),
    ]

    # matrix_transpose: matrix_transpose(x) - 2D fp16
    cases["matrix_transpose"] = [
        ("small", "matrix_transpose",
         (torch.randn(1024, 1024, device="cuda", dtype=torch.float16),),
         None, "x:(1024,1024) fp16", False),
        ("large", "matrix_transpose",
         (torch.randn(16384, 16384, device="cuda", dtype=torch.float16),),
         None, "x:(16384,16384) fp16", False),
    ]

    # max_reduction: max_reduction(input, dim) - reduce dim=1
    cases["max_reduction"] = [
        ("small", "max_reduction",
         (torch.randn(1024, 1024, device="cuda", dtype=torch.float16), 1),
         None, "x:(1024,1024) dim=1 fp16", False),
        ("large", "max_reduction",
         (torch.randn(8192, 32768, device="cuda", dtype=torch.float16), 1),
         None, "x:(8192,32768) dim=1 fp16", False),
    ]

    # mean_reduction: mean_reduction(input_tensor, dim) - fp32
    cases["mean_reduction"] = [
        ("small", "mean_reduction",
         (torch.randn(1024, 1024, device="cuda", dtype=torch.float32), 1),
         None, "x:(1024,1024) dim=1 fp32", False),
        ("large", "mean_reduction",
         (torch.randn(8192, 32768, device="cuda", dtype=torch.float32), 1),
         None, "x:(8192,32768) dim=1 fp32", False),
    ]

    # mul: mul(x) = 2*x
    cases["mul"] = [
        ("small", "mul",
         (torch.randn(4096, device="cuda", dtype=torch.float16),),
         None, "x:(4096,) fp16", False),
        ("large", "mul",
         (torch.randn(M64, device="cuda", dtype=torch.float16),),
         None, "x:(64M,) fp16", False),
    ]

    # relu: relu(x)
    cases["relu"] = [
        ("small", "relu",
         (torch.randn(4096, 4096, device="cuda", dtype=torch.float16),),
         None, "x:(4096,4096) fp16", False),
        ("large", "relu",
         (torch.randn(16384, 16384, device="cuda", dtype=torch.float16),),
         None, "x:(16384,16384) fp16", False),
    ]

    # rms_norm: rms_norm(x, normalized_shape, weight)
    cases["rms_norm"] = [
        ("small", "rms_norm",
         (torch.randn(512, 1024, device="cuda", dtype=torch.float16),
          (1024,),
          torch.randn(1024, device="cuda", dtype=torch.float16)),
         None, "x:(512,1024) w:(1024,) fp16", False),
        ("large", "rms_norm",
         (torch.randn(8192, 8192, device="cuda", dtype=torch.float16),
          (8192,),
          torch.randn(8192, device="cuda", dtype=torch.float16)),
         None, "x:(8192,8192) w:(8192,) fp16", False),
    ]

    # softmax: softmax(x) - row-wise
    cases["softmax"] = [
        ("small", "softmax",
         (torch.randn(512, 1024, device="cuda", dtype=torch.float16),),
         None, "x:(512,1024) fp16", False),
        ("large", "softmax",
         (torch.randn(4096, 32768, device="cuda", dtype=torch.float16),),
         None, "x:(4096,32768) fp16", False),
    ]

    # swiglu: swiglu(xy) where xy is concatenated gate and up
    cases["swiglu"] = [
        ("small", "swiglu",
         (torch.randn(512, 8192, device="cuda", dtype=torch.float16),),
         None, "xy:(512,8192) fp16", False),
        ("large", "swiglu",
         (torch.randn(4096, 32768, device="cuda", dtype=torch.float16),),
         None, "xy:(4096,32768) fp16", False),
    ]

    return cases


def main():
    csv_path = RESULTS_DIR / "profile.csv"
    rows = []
    cases = get_test_cases()

    kernel_dirs = sorted(
        d for d in BENCH_DIR.iterdir()
        if d.is_dir() and (d / "pytorch_impl.py").exists()
    )

    total_cases = sum(len(v) for v in cases.values())
    print(f"Profiling {len(kernel_dirs)} kernels, {total_cases} test cases "
          f"({WARMUP_ITERS} warmup, {MEASURE_ITERS}/{MEASURE_ITERS_SLOW} measure)...\n")
    print(f"{'Kernel':<22} {'Size':<6} {'Impl':<10} {'Latency(ms)':>12} {'PeakMem(MB)':>12}  Input")
    print("-" * 100)

    for kdir in kernel_dirs:
        name = kdir.name
        if name not in cases:
            print(f"  SKIP {name}: no test cases defined")
            continue

        try:
            pt_mod = load_module(kdir, "pytorch_impl")
            tr_mod = load_module(kdir, "triton_impl")
        except Exception:
            print(f"  ERROR {name}: failed to import")
            traceback.print_exc()
            continue

        for (size_label, fn_name, args, kwargs, desc, is_slow) in cases[name]:
            kwargs = kwargs or {}
            iters = MEASURE_ITERS_SLOW if is_slow else MEASURE_ITERS
            warmup = 3 if is_slow else WARMUP_ITERS

            try:
                pt_fn = getattr(pt_mod, fn_name)
                tr_fn = getattr(tr_mod, fn_name)
            except AttributeError as e:
                print(f"  ERROR {name}/{size_label}: {e}")
                continue

            # Profile PyTorch
            try:
                torch.cuda.empty_cache()
                pt_lat, pt_mem = profile_fn(pt_fn, args, kwargs, warmup=warmup, iters=iters)
                print(f"{name:<22} {size_label:<6} {'pytorch':<10} {pt_lat:>12.4f} {pt_mem:>12.2f}  {desc}")
                rows.append({
                    "kernel": name, "size": size_label, "impl": "pytorch",
                    "input_desc": desc,
                    "latency_ms": round(pt_lat, 4),
                    "peak_memory_mb": round(pt_mem, 2),
                })
            except Exception as e:
                err_msg = str(e)[:80]
                print(f"{name:<22} {size_label:<6} {'pytorch':<10} {'ERROR':>12}  {err_msg}")
                rows.append({
                    "kernel": name, "size": size_label, "impl": "pytorch",
                    "input_desc": desc,
                    "latency_ms": "ERROR",
                    "peak_memory_mb": "ERROR",
                })

            # Profile Triton
            try:
                torch.cuda.empty_cache()
                tr_lat, tr_mem = profile_fn(tr_fn, args, kwargs, warmup=warmup, iters=iters)
                print(f"{'':<22} {'':<6} {'triton':<10} {tr_lat:>12.4f} {tr_mem:>12.2f}")
                rows.append({
                    "kernel": name, "size": size_label, "impl": "triton",
                    "input_desc": desc,
                    "latency_ms": round(tr_lat, 4),
                    "peak_memory_mb": round(tr_mem, 2),
                })
            except Exception as e:
                err_msg = str(e)[:80]
                print(f"{'':<22} {'':<6} {'triton':<10} {'ERROR':>12}  {err_msg}")
                rows.append({
                    "kernel": name, "size": size_label, "impl": "triton",
                    "input_desc": desc,
                    "latency_ms": "ERROR",
                    "peak_memory_mb": "ERROR",
                })

            # Speedup
            if (len(rows) >= 2
                    and rows[-2].get("latency_ms") != "ERROR"
                    and rows[-1].get("latency_ms") != "ERROR"):
                speedup = rows[-2]["latency_ms"] / rows[-1]["latency_ms"] if rows[-1]["latency_ms"] > 0 else float("inf")
                print(f"{'':<22} {'':<6} {'speedup':<10} {speedup:>12.2f}x")

            print()

    # Write CSV
    fieldnames = ["kernel", "size", "impl", "input_desc", "latency_ms", "peak_memory_mb"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
