#!/usr/bin/env python3
"""Re-profile only the 3 fixed kernels and patch profile.csv."""
import csv
import importlib
import time
import traceback
from pathlib import Path
import torch

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"
WARMUP_ITERS = 10
MEASURE_ITERS = 100
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
    spec = importlib.util.spec_from_file_location(module_name, str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cases = {
    "batched_matmul": [
        ("small", "batched_matmul",
         (torch.randn(64, 128, device="cuda", dtype=torch.float16),
          torch.randn(64, 128, 128, device="cuda", dtype=torch.float16)),
         None, "A:(64,128) B:(64,128,128) fp16", False),
        ("large", "batched_matmul",
         (torch.randn(128, 2048, device="cuda", dtype=torch.float16),
          torch.randn(128, 2048, 2048, device="cuda", dtype=torch.float16)),
         None, "A:(128,2048) B:(128,2048,2048) fp16", False),
    ],
    "embedding": [
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
    ],
    "softmax": [
        ("small", "softmax",
         (torch.randn(512, 1024, device="cuda", dtype=torch.float16),),
         None, "x:(512,1024) fp16", False),
        ("large", "softmax",
         (torch.randn(4096, 32768, device="cuda", dtype=torch.float16),),
         None, "x:(4096,32768) fp16", False),
    ],
}

new_results = {}
for name, tcases in cases.items():
    kdir = BENCH_DIR / name
    try:
        tl_mod = load_module(kdir, "tilelang_impl")
    except Exception:
        print(f"ERROR importing {name}")
        traceback.print_exc()
        continue

    for (size_label, fn_name, args, kwargs, desc, is_slow) in tcases:
        kwargs = kwargs or {}
        iters = 5 if is_slow else MEASURE_ITERS
        warmup = 3 if is_slow else WARMUP_ITERS
        try:
            tl_fn = getattr(tl_mod, fn_name)
            torch.cuda.empty_cache()
            lat, mem = profile_fn(tl_fn, args, kwargs, warmup=warmup, iters=iters)
            print(f"{name:<22} {size_label:<6} tilelang {lat:>12.4f} {mem:>12.2f}  {desc}")
            new_results[(name, size_label)] = {
                "kernel": name, "size": size_label, "impl": "tilelang",
                "input_desc": desc,
                "latency_ms": round(lat, 4),
                "peak_memory_mb": round(mem, 2),
            }
        except Exception as e:
            print(f"{name:<22} {size_label:<6} tilelang ERROR  {e}")
            new_results[(name, size_label)] = {
                "kernel": name, "size": size_label, "impl": "tilelang",
                "input_desc": desc,
                "latency_ms": "ERROR",
                "peak_memory_mb": "ERROR",
            }

# Patch CSV
csv_path = RESULTS_DIR / "profile.csv"
rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["kernel"], row["size"])
        if row["impl"] == "tilelang" and key in new_results:
            rows.append(new_results[key])
        else:
            rows.append(row)

fieldnames = ["kernel", "size", "impl", "input_desc", "latency_ms", "peak_memory_mb"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nPatched {csv_path}")
