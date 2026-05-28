#!/usr/bin/env python3
"""
Profile tuned Triton and TileLang kernels.
Appends triton_tuned / tilelang_tuned rows to ViperBench/results/profile.csv.
Modules are imported fresh so they pick up tuning_cache.json at load time.
"""
import csv
import importlib
import importlib.util
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
MEASURE_ITERS_SLOW = 5

M64 = 64 * 1024 * 1024


def profile_fn(fn, args, kwargs=None, warmup=WARMUP_ITERS, iters=MEASURE_ITERS):
    kwargs = kwargs or {}
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

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

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)

    return median_ms, peak_mem_mb


def load_module(kernel_dir, module_name):
    mod_path = kernel_dir / f"{module_name}.py"
    name = f"{kernel_dir.name}_{module_name}_tuned"
    spec = importlib.util.spec_from_file_location(name, str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_test_cases():
    """Same test cases as benchmark.py / benchmark_tilelang.py."""
    cases = {}

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

    cases["argmax"] = [
        ("small", "argmax",
         (torch.randn(1024, 1024, device="cuda", dtype=torch.float16), 1),
         None, "x:(1024,1024) dim=1 fp16", False),
        ("large", "argmax",
         (torch.randn(8192, 32768, device="cuda", dtype=torch.float16), 1),
         None, "x:(8192,32768) dim=1 fp16", False),
    ]

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

    cases["log_softmax"] = [
        ("small", "log_softmax",
         (torch.randn(512, 1024, device="cuda", dtype=torch.float16),),
         None, "x:(512,1024) fp16", False),
        ("large", "log_softmax",
         (torch.randn(4096, 32768, device="cuda", dtype=torch.float16),),
         None, "x:(4096,32768) fp16", False),
    ]

    cases["logsumexp"] = [
        ("small", "logsumexp",
         (torch.randn(512, 1024, device="cuda", dtype=torch.float32),),
         None, "x:(512,1024) fp32", False),
        ("large", "logsumexp",
         (torch.randn(8192, 16384, device="cuda", dtype=torch.float32),),
         None, "x:(8192,16384) fp32", False),
    ]

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

    cases["matrix_transpose"] = [
        ("small", "matrix_transpose",
         (torch.randn(1024, 1024, device="cuda", dtype=torch.float16),),
         None, "x:(1024,1024) fp16", False),
        ("large", "matrix_transpose",
         (torch.randn(16384, 16384, device="cuda", dtype=torch.float16),),
         None, "x:(16384,16384) fp16", False),
    ]

    cases["max_reduction"] = [
        ("small", "max_reduction",
         (torch.randn(1024, 1024, device="cuda", dtype=torch.float16), 1),
         None, "x:(1024,1024) dim=1 fp16", False),
        ("large", "max_reduction",
         (torch.randn(8192, 32768, device="cuda", dtype=torch.float16), 1),
         None, "x:(8192,32768) dim=1 fp16", False),
    ]

    cases["mean_reduction"] = [
        ("small", "mean_reduction",
         (torch.randn(1024, 1024, device="cuda", dtype=torch.float32), 1),
         None, "x:(1024,1024) dim=1 fp32", False),
        ("large", "mean_reduction",
         (torch.randn(8192, 32768, device="cuda", dtype=torch.float32), 1),
         None, "x:(8192,32768) dim=1 fp32", False),
    ]

    cases["mul"] = [
        ("small", "mul",
         (torch.randn(4096, device="cuda", dtype=torch.float16),),
         None, "x:(4096,) fp16", False),
        ("large", "mul",
         (torch.randn(M64, device="cuda", dtype=torch.float16),),
         None, "x:(64M,) fp16", False),
    ]

    cases["relu"] = [
        ("small", "relu",
         (torch.randn(4096, 4096, device="cuda", dtype=torch.float16),),
         None, "x:(4096,4096) fp16", False),
        ("large", "relu",
         (torch.randn(16384, 16384, device="cuda", dtype=torch.float16),),
         None, "x:(16384,16384) fp16", False),
    ]

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

    cases["softmax"] = [
        ("small", "softmax",
         (torch.randn(512, 1024, device="cuda", dtype=torch.float16),),
         None, "x:(512,1024) fp16", False),
        ("large", "softmax",
         (torch.randn(4096, 32768, device="cuda", dtype=torch.float16),),
         None, "x:(4096,32768) fp16", False),
    ]

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
    # Verify tuning cache exists
    cache_path = BENCH_DIR / "results" / "tuning_cache.json"
    if not cache_path.exists():
        print(f"ERROR: tuning cache not found at {cache_path}")
        print("Run tuning sweep first to generate tuning_cache.json")
        sys.exit(1)

    import json
    cache = json.loads(cache_path.read_text())
    print(f"Tuning cache loaded: {len(cache)} entries")

    csv_path = RESULTS_DIR / "profile.csv"

    # Read existing rows, strip old tuned rows
    existing_rows = []
    fieldnames = ["kernel", "size", "impl", "input_desc", "latency_ms", "peak_memory_mb"]
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("impl") not in ("triton_tuned", "tilelang_tuned"):
                    existing_rows.append(row)

    new_rows = []
    cases = get_test_cases()

    kernel_dirs = sorted(
        d for d in BENCH_DIR.iterdir()
        if d.is_dir() and (d / "triton_impl.py").exists()
    )

    total = sum(len(v) for v in cases.values())
    print(f"\nProfiling {len(kernel_dirs)} kernels (tuned), {total} test cases "
          f"({WARMUP_ITERS} warmup, {MEASURE_ITERS}/{MEASURE_ITERS_SLOW} measure)...\n")
    print(f"{'Kernel':<22} {'Size':<6} {'Impl':<16} {'Latency(ms)':>12} {'PeakMem(MB)':>12}  Input")
    print("-" * 110)

    for kdir in kernel_dirs:
        name = kdir.name
        if name not in cases:
            print(f"  SKIP {name}: no test cases defined")
            continue

        # Load modules (they will pick up tuning cache automatically)
        tr_mod = None
        tl_mod = None
        try:
            tr_mod = load_module(kdir, "triton_impl")
        except Exception:
            print(f"  ERROR {name}: failed to import triton_impl")
            traceback.print_exc()
        try:
            tl_mod = load_module(kdir, "tilelang_impl")
        except Exception:
            print(f"  ERROR {name}: failed to import tilelang_impl")
            traceback.print_exc()

        if not tr_mod and not tl_mod:
            continue

        for (size_label, fn_name, args, kwargs, desc, is_slow) in cases[name]:
            kwargs = kwargs or {}
            iters = MEASURE_ITERS_SLOW if is_slow else MEASURE_ITERS
            warmup = 3 if is_slow else WARMUP_ITERS

            # Profile tuned Triton
            if tr_mod:
                try:
                    tr_fn = getattr(tr_mod, fn_name)
                    torch.cuda.empty_cache()
                    tr_lat, tr_mem = profile_fn(tr_fn, args, kwargs, warmup=warmup, iters=iters)
                    print(f"{name:<22} {size_label:<6} {'triton_tuned':<16} {tr_lat:>12.4f} {tr_mem:>12.2f}  {desc}")
                    new_rows.append({
                        "kernel": name, "size": size_label, "impl": "triton_tuned",
                        "input_desc": desc,
                        "latency_ms": round(tr_lat, 4),
                        "peak_memory_mb": round(tr_mem, 2),
                    })
                except Exception as e:
                    err_msg = str(e)[:80]
                    print(f"{name:<22} {size_label:<6} {'triton_tuned':<16} {'ERROR':>12}  {err_msg}")
                    new_rows.append({
                        "kernel": name, "size": size_label, "impl": "triton_tuned",
                        "input_desc": desc, "latency_ms": "ERROR", "peak_memory_mb": "ERROR",
                    })

            # Profile tuned TileLang
            if tl_mod:
                try:
                    tl_fn = getattr(tl_mod, fn_name)
                    torch.cuda.empty_cache()
                    tl_lat, tl_mem = profile_fn(tl_fn, args, kwargs, warmup=warmup, iters=iters)
                    print(f"{'':<22} {'':<6} {'tilelang_tuned':<16} {tl_lat:>12.4f} {tl_mem:>12.2f}")
                    new_rows.append({
                        "kernel": name, "size": size_label, "impl": "tilelang_tuned",
                        "input_desc": desc,
                        "latency_ms": round(tl_lat, 4),
                        "peak_memory_mb": round(tl_mem, 2),
                    })
                except Exception as e:
                    err_msg = str(e)[:80]
                    print(f"{'':<22} {'':<6} {'tilelang_tuned':<16} {'ERROR':>12}  {err_msg}")
                    new_rows.append({
                        "kernel": name, "size": size_label, "impl": "tilelang_tuned",
                        "input_desc": desc, "latency_ms": "ERROR", "peak_memory_mb": "ERROR",
                    })

        print()

    # Merge: insert tuned rows after their untuned counterparts
    merged = []
    new_by_key = {}
    for row in new_rows:
        key = (row["kernel"], row["size"], row["impl"])
        new_by_key[key] = row

    for row in existing_rows:
        merged.append(row)
        k, s, impl = row["kernel"], row["size"], row["impl"]
        # Insert triton_tuned after triton
        if impl == "triton":
            tuned_key = (k, s, "triton_tuned")
            if tuned_key in new_by_key:
                merged.append(new_by_key.pop(tuned_key))
        # Insert tilelang_tuned after tilelang
        elif impl == "tilelang":
            tuned_key = (k, s, "tilelang_tuned")
            if tuned_key in new_by_key:
                merged.append(new_by_key.pop(tuned_key))

    # Append any remaining tuned rows not yet inserted
    for row in new_by_key.values():
        merged.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    n_tuned = len(new_rows)
    print(f"\nResults written to {csv_path} ({len(merged)} total rows: "
          f"{len(existing_rows)} existing + {n_tuned} tuned)")


if __name__ == "__main__":
    main()
