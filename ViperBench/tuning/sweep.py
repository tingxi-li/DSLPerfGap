#!/usr/bin/env python3
"""
One-time auto-tuning sweep for ViperBench kernels.

Usage:
    python -m tuning.sweep --all                    # sweep all kernels
    python -m tuning.sweep --kernel matmul --impl triton
    python -m tuning.sweep --kernel layer_norm --impl tilelang
"""
import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

import torch

BENCH_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BENCH_DIR))

import tuning.cache as _cachemod
from tuning.cache import get_gpu_arch, load_cache, save_cache
from tuning.configs import TRITON_CONFIGS, TILELANG_CONFIGS

# Rigor matched to the benchmark harness: enough warmup to absorb the
# Triton/TileLang JIT + autotune, CUDA-event timing (not wall-clock) so the
# selection is not corrupted by compile tails or host-launch jitter.
WARMUP = 15
TRIALS = 50


def time_fn(fn, args, warmup=WARMUP, trials=TRIALS):
    """Median latency in ms, measured with CUDA events on an idle GPU."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(trials):
        starter.record()
        fn(*args)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms
    times.sort()
    return times[len(times) // 2]


def load_module(kernel_dir, module_name):
    """Import a module from a kernel directory."""
    mod_path = kernel_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"{kernel_dir.name}_{module_name}", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Input generators per kernel (large size from benchmark.py) ──

M64 = 64 * 1024 * 1024

def _make_inputs(kernel_name):
    """Generate benchmark inputs for a kernel."""
    generators = {
        "add": lambda: (torch.randn(M64, device="cuda", dtype=torch.float16),
                        torch.randn(M64, device="cuda", dtype=torch.float16)),
        "mul": lambda: (torch.randn(M64, device="cuda", dtype=torch.float16),),
        "relu": lambda: (torch.randn(4096, 4096, device="cuda", dtype=torch.float16),),
        "argmax": lambda: (torch.randn(1024, 1024, device="cuda", dtype=torch.float16), 1),
        "matmul": lambda: (torch.randn(4096, 4096, device="cuda", dtype=torch.float16),
                           torch.randn(4096, 4096, device="cuda", dtype=torch.float16)),
        "leaky_relu": lambda: (torch.randn(4096, 4096, device="cuda", dtype=torch.float16),
                               torch.randn(4096, 4096, device="cuda", dtype=torch.float16),
                               "leaky_relu"),
        "batched_matmul": lambda: (torch.randn(64, 128, device="cuda", dtype=torch.float16),
                                   torch.randn(64, 128, 128, device="cuda", dtype=torch.float16)),
        "conv2d": lambda: (torch.randn(8, 64, 56, 56, device="cuda", dtype=torch.float16),
                           torch.randn(64, 64, 3, 3, device="cuda", dtype=torch.float16)),
        "embedding": lambda: (torch.randint(0, 8192, (2048,), device="cuda", dtype=torch.int32),
                              torch.randn(8192, 256, device="cuda", dtype=torch.float16),
                              0, 8192,
                              torch.zeros(2048, 256, device="cuda", dtype=torch.float16)),
        "index_select": lambda: (torch.empty(256, 512, device="cuda", dtype=torch.float16),
                                 torch.randn(4096, 512, device="cuda", dtype=torch.float16),
                                 torch.randint(0, 4096, (256,), device="cuda", dtype=torch.int64)),
        "layer_norm": lambda: (torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16),
                               torch.randn(1024, device="cuda", dtype=torch.bfloat16),
                               torch.randn(1024, device="cuda", dtype=torch.bfloat16)),
        "linear_activation": lambda: (torch.randn(1, 256, 1024, device="cuda", dtype=torch.float16),
                                      torch.randn(4096, 1024, device="cuda", dtype=torch.float16),
                                      torch.randn(4096, 1024, device="cuda", dtype=torch.float16),
                                      torch.randn(1024, device="cuda", dtype=torch.float16)),
        "log_softmax": lambda: (torch.randn(512, 1024, device="cuda", dtype=torch.float16),),
        "logsumexp": lambda: (torch.randn(512, 1024, device="cuda", dtype=torch.float32),),
        "matrix_transpose": lambda: (torch.randn(1024, 1024, device="cuda", dtype=torch.float16),),
        "max_reduction": lambda: (torch.randn(1024, 1024, device="cuda", dtype=torch.float16), 1),
        "mean_reduction": lambda: (torch.randn(1024, 1024, device="cuda", dtype=torch.float32), 1),
        "rms_norm": lambda: (torch.randn(512, 1024, device="cuda", dtype=torch.float16),
                             (1024,),
                             torch.randn(1024, device="cuda", dtype=torch.float16)),
        "softmax": lambda: (torch.randn(512, 1024, device="cuda", dtype=torch.float16),),
        "swiglu": lambda: (torch.randn(512, 8192, device="cuda", dtype=torch.float16),),
        "cross_entropy": lambda: (torch.randn(256, 1024, device="cuda", dtype=torch.float32),
                                  torch.randint(0, 1024, (256,), device="cuda", dtype=torch.int64),
                                  0.0, 1.0, 0.0, -100, 1024, 0, 256, False, False),
        "attention": lambda: (torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.float32),
                              torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.float32),
                              torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.float32)),
    }
    if kernel_name not in generators:
        return None
    return generators[kernel_name]()


# ── Function name mapping ──

KERNEL_FN_NAMES = {
    "add": "add", "mul": "mul", "relu": "relu", "argmax": "argmax",
    "matmul": "matmul", "leaky_relu": "leaky_relu",
    "batched_matmul": "batched_matmul", "conv2d": "conv2d",
    "embedding": "embedding", "index_select": "index_select",
    "layer_norm": "layer_norm", "linear_activation": "kernel_ff",
    "log_softmax": "log_softmax", "logsumexp": "logsumexp",
    "matrix_transpose": "matrix_transpose", "max_reduction": "max_reduction",
    "mean_reduction": "mean_reduction", "rms_norm": "rms_norm",
    "softmax": "softmax", "swiglu": "swiglu",
    "cross_entropy": "cross_entropy_fwd", "attention": "attention_fwd",
}


def sweep_kernel(kernel_name, impl_type, configs):
    """Sweep configs for a single kernel. Returns best (config, latency)."""
    kdir = BENCH_DIR / kernel_name
    module_name = f"{impl_type}_impl"

    inputs = _make_inputs(kernel_name)
    if inputs is None:
        print(f"  SKIP {kernel_name}: no input generator")
        return None, float("inf")

    fn_name = KERNEL_FN_NAMES.get(kernel_name)
    if fn_name is None:
        print(f"  SKIP {kernel_name}: unknown function name")
        return None, float("inf")

    # Candidate set includes the impl's hardcoded default (config=None -> the
    # cache-miss path) so the sweep can never select a grid config that is
    # worse than the default at the tuning shape.
    candidates = [None] + list(configs)
    best_config = None
    best_latency = float("inf")

    # Each candidate must be applied at BUILD time. The impls read _TUNED at
    # import (via get_best_config) to construct the kernel, so setting
    # mod._TUNED AFTER load_module is a no-op (the kernel is already built).
    # Instead, patch get_best_config so the *fresh import* builds the kernel
    # with this candidate, then restore it.
    _orig_gbc = _cachemod.get_best_config
    for i, config in enumerate(candidates):
        label = "default" if config is None else str(config)
        try:
            _cachemod.get_best_config = (lambda *a, _c=config, **k: _c)
            mod = load_module(kdir, module_name)  # builds kernel with `config`
            fn = getattr(mod, fn_name)
            torch.cuda.empty_cache()

            latency = time_fn(fn, inputs)
            status = f"{latency:.4f}ms"

            if latency < best_latency:
                best_latency = latency
                best_config = config

        except Exception as e:
            status = f"ERROR: {str(e)[:60]}"
        finally:
            _cachemod.get_best_config = _orig_gbc

        print(f"  [{i+1}/{len(candidates)}] {label} -> {status}")

    return best_config, best_latency


def run_sweep(kernel_name=None, impl_type=None):
    """Run sweep for specified or all kernels."""
    cache = load_cache()
    arch = get_gpu_arch()

    if kernel_name and impl_type:
        pairs = [(kernel_name, impl_type)]
    else:
        # All kernels, both impls
        all_kernels = sorted(set(list(TRITON_CONFIGS.keys()) + list(TILELANG_CONFIGS.keys())))
        pairs = []
        for k in all_kernels:
            if k in TRITON_CONFIGS:
                pairs.append((k, "triton"))
            if k in TILELANG_CONFIGS:
                pairs.append((k, "tilelang"))

    for kname, itype in pairs:
        configs_map = TRITON_CONFIGS if itype == "triton" else TILELANG_CONFIGS
        configs = configs_map.get(kname, [])
        if not configs:
            print(f"\n=== {kname} ({itype}): no configs defined, skip ===")
            continue

        print(f"\n{'='*60}")
        print(f"  Sweeping: {kname} ({itype}) — {len(configs)} configs")
        print(f"{'='*60}")

        best_config, best_latency = sweep_kernel(kname, itype, configs)

        key = f"{kname}/{itype}/{arch}"
        if best_config is not None:
            cache[key] = best_config
            print(f"  BEST: {best_config} -> {best_latency:.4f}ms")
        else:
            # The hardcoded default beat every grid config at the tuning shape;
            # do not write an override (and drop any stale entry) so the impl
            # falls back to its default. This makes Delta=0 the honest result.
            cache.pop(key, None)
            print(f"  BEST: default (no grid config beat it) -> "
                  f"{best_latency:.4f}ms; no cache override written")

    save_cache(cache)
    print(f"\nCache saved to {cache_path()}")


def cache_path():
    from tuning.cache import CACHE_PATH
    return CACHE_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-tuning sweep for ViperBench")
    parser.add_argument("--kernel", help="Kernel name (e.g., matmul)")
    parser.add_argument("--impl", choices=["triton", "tilelang"], help="Implementation type")
    parser.add_argument("--all", action="store_true", help="Sweep all kernels")
    args = parser.parse_args()

    if args.all:
        run_sweep()
    elif args.kernel and args.impl:
        run_sweep(args.kernel, args.impl)
    else:
        parser.error("Specify --all or both --kernel and --impl")
