"""
ViperBench evaluation runner — main orchestration loop.

Usage:
    python -m viperbench.runner --kernel matmul --sweep prioritized
    python -m viperbench.runner --all --sweep prioritized
    python -m viperbench.runner --kernel matmul --correctness-only
    python -m viperbench.runner --list
    python -m viperbench.runner --kernel matmul --hardware configs/hardware/a100_80gb.json
    python -m viperbench.runner --kernel matmul --shape prefill --dtype fp16
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .utils import (
    RESULTS_DIR,
    detect_hardware,
    get_torch_dtype,
    import_module_from_path,
    load_hardware_config,
    setup_logger,
)
from .validate import check_correctness, ValidationResult
from .profile import profile_latency, profile_memory, LatencyResult, MemoryResult
from .metrics import compute_flops, compute_bytes, compute_sol, resolve_metrics, SOLResult
from .input_gen import generate, resolve_input_gen

KERNELS_DIR = Path(__file__).resolve().parent.parent / "kernels"


def _deep_copy_inputs(inputs):
    """Deep-copy input dict so implementations can't mutate shared tensors."""
    copied = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            copied[k] = v.clone()
        else:
            copied[k] = v
    return copied
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Implementation file patterns
IMPL_PATTERNS = {
    "pytorch_eager":   "reference.py",
    "pytorch_compile": "reference.py",
    "triton_impl":     "triton_impl.py",
    "tilelang_impl":   "tilelang_impl.py",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    kernel: str
    config_name: str
    config: Dict[str, Any]
    implementation: str

    # Correctness
    correct: Optional[bool] = None
    max_abs_error: Optional[float] = None
    max_rel_error: Optional[float] = None
    mismatch_fraction: Optional[float] = None
    error_message: Optional[str] = None

    # Latency
    median_us: Optional[float] = None
    mean_us: Optional[float] = None
    min_us: Optional[float] = None
    max_us: Optional[float] = None
    std_us: Optional[float] = None

    # Memory
    peak_allocated_mb: Optional[float] = None
    peak_reserved_mb: Optional[float] = None
    overhead_mb: Optional[float] = None

    # Derived metrics
    achieved_tflops: Optional[float] = None
    sol_compute_pct: Optional[float] = None
    achieved_bw_gb_s: Optional[float] = None
    sol_memory_pct: Optional[float] = None
    bottleneck: Optional[str] = None

    # torch.compile specific
    compile_time_s: Optional[float] = None

    # Metadata
    hardware: str = ""
    dtype: str = ""
    timestamp: str = ""
    status: str = "ok"  # ok, not_implemented, import_error, runtime_error, oom


# ---------------------------------------------------------------------------
# Sweep generation
# ---------------------------------------------------------------------------

def _generate_sweep_configs(input_config: dict, sweep_mode: str,
                            shape_filter: Optional[str] = None,
                            dtype_filter: Optional[str] = None,
                            extra_filters: Optional[Dict[str, str]] = None,
                            ) -> List[Dict[str, Any]]:
    """Generate list of config dicts from input_config.json sweep settings."""
    params = input_config.get("params", {})
    shapes = params.get("shape", [{}])
    dtypes = params.get("dtype", ["fp32"])

    # Determine which sweep to use
    if sweep_mode == "prioritized":
        sweep_spec = input_config.get("priority_sweep", {})
    elif sweep_mode == "full":
        sweep_spec = input_config.get("full_sweep", {})
    else:
        sweep_spec = input_config.get("priority_sweep", {})

    # Collect axes to sweep
    axes = {}

    # Shape axis
    if shape_filter:
        axes["shape"] = [s for s in shapes if s.get("name") == shape_filter]
        if not axes["shape"]:
            axes["shape"] = shapes[:1]  # fallback to first
    elif sweep_spec.get("shape") == "all":
        axes["shape"] = shapes
    elif isinstance(sweep_spec.get("shape"), list):
        names = sweep_spec["shape"]
        axes["shape"] = [s for s in shapes if s.get("name") in names]
    else:
        axes["shape"] = shapes

    # Dtype axis
    if dtype_filter:
        axes["dtype"] = [dtype_filter] if dtype_filter in dtypes else dtypes[:1]
    elif sweep_spec.get("dtype") == "all":
        axes["dtype"] = dtypes
    elif isinstance(sweep_spec.get("dtype"), list):
        axes["dtype"] = sweep_spec["dtype"]
    else:
        axes["dtype"] = dtypes

    # Other param axes (transpose, structure, stride, padding, etc.)
    extra_axes = {}
    for key, values in params.items():
        if key in ("shape", "dtype"):
            continue
        if not isinstance(values, list):
            continue
        spec_val = sweep_spec.get(key)
        if extra_filters and key in extra_filters:
            extra_axes[key] = [extra_filters[key]]
        elif spec_val == "all":
            extra_axes[key] = values
        elif isinstance(spec_val, list):
            extra_axes[key] = spec_val
        else:
            extra_axes[key] = values[:1]  # default to first value

    # Generate Cartesian product
    configs = []
    extra_keys = sorted(extra_axes.keys())
    extra_values_lists = [extra_axes[k] for k in extra_keys]

    if extra_values_lists:
        extra_combos = list(product(*extra_values_lists))
    else:
        extra_combos = [()]

    for shape_cfg in axes["shape"]:
        for dtype_str in axes["dtype"]:
            for extra_combo in extra_combos:
                cfg = dict(shape_cfg)  # copy shape params
                cfg["dtype"] = dtype_str
                for k, v in zip(extra_keys, extra_combo):
                    cfg[k] = v
                configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# Implementation discovery and loading
# ---------------------------------------------------------------------------

def _discover_implementations(kernel_dir: Path) -> Dict[str, Path]:
    """Discover implementation files in a kernel directory."""
    impls = {}
    for impl_name, filename in IMPL_PATTERNS.items():
        fpath = kernel_dir / filename
        if fpath.exists():
            impls[impl_name] = fpath

    # Also discover additional *_impl.py files
    for p in kernel_dir.glob("*_impl.py"):
        name = p.stem
        if name not in impls:
            impls[name] = p

    return impls


def _load_impl_callable(impl_name: str, impl_path: Path):
    """Load and return the callable for an implementation.
    Returns (callable, is_not_implemented, module) or raises on error.
    """
    module = import_module_from_path(impl_path)

    if getattr(module, "NOT_IMPLEMENTED", False):
        return None, True, module

    if impl_name in ("pytorch_eager", "pytorch_compile"):
        fn = module.reference
    elif hasattr(module, "kernel"):
        fn = module.kernel
    elif hasattr(module, "reference"):
        fn = module.reference
    else:
        raise AttributeError(
            "Implementation %s has no 'reference' or 'kernel' function" % impl_path
        )

    return fn, False, module


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def _evaluate_single(
    kernel_name: str,
    config: Dict[str, Any],
    impl_name: str,
    fn: Callable,
    golden_outputs: Dict[str, torch.Tensor],
    inputs: Dict[str, Any],
    input_config: dict,
    hardware: dict,
    category: str,
    kernel_dir: Path,
    correctness_only: bool,
    logger,
) -> EvalResult:
    """Evaluate a single (kernel, config, implementation) tuple."""
    dtype_str = config.get("dtype", "fp32")
    config_name = config.get("name", "default")
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    hw_name = hardware.get("name", "unknown")

    result = EvalResult(
        kernel=kernel_name,
        config_name=config_name,
        config=config,
        implementation=impl_name,
        hardware=hw_name,
        dtype=dtype_str,
        timestamp=timestamp,
    )

    # Get tolerances
    correctness_cfg = input_config.get("correctness", {})
    atol_map = correctness_cfg.get("atol", {})
    rtol_map = correctness_cfg.get("rtol", {})
    atol = atol_map.get(dtype_str, 1e-5)
    rtol = rtol_map.get(dtype_str, 1e-5)

    # --- Correctness ---
    try:
        if impl_name == "pytorch_compile":
            compiled_fn = torch.compile(fn, mode="reduce-overhead")
            # Warmup compile
            for _ in range(3):
                compiled_fn(inputs)
            torch.cuda.synchronize()
            test_fn = compiled_fn
        else:
            test_fn = fn

        test_outputs = test_fn(inputs)
        val_result = check_correctness(golden_outputs, test_outputs, atol, rtol)

        result.correct = val_result.passed
        if val_result.per_tensor:
            worst = max(val_result.per_tensor.values(),
                        key=lambda tc: tc.max_abs_error)
            result.max_abs_error = worst.max_abs_error
            result.max_rel_error = worst.max_rel_error
            result.mismatch_fraction = worst.mismatch_fraction
        if not val_result.passed:
            result.error_message = val_result.error_message
            result.status = "correctness_failed"
            logger.warning("  %s correctness FAIL: %s",
                           impl_name, (val_result.error_message or "")[:120])
            return result

    except torch.cuda.OutOfMemoryError as e:
        result.status = "oom"
        result.correct = None
        result.error_message = str(e)[:200]
        logger.warning("  %s OOM: %s", impl_name, result.error_message[:80])
        return result
    except Exception as e:
        result.status = "runtime_error"
        result.correct = None
        result.error_message = "%s: %s" % (type(e).__name__, str(e)[:200])
        logger.warning("  %s runtime error: %s", impl_name, result.error_message[:80])
        return result

    if correctness_only:
        return result

    # --- Profiling ---
    profiling_cfg = input_config.get("profiling", {})
    warmup = profiling_cfg.get("warmup_iters", 10)
    timed = profiling_cfg.get("timed_iters", 100)
    clear_l2 = profiling_cfg.get("clear_l2_cache", True)

    # Get L2 cache size from hardware
    mem_hier = hardware.get("memory_hierarchy", {})
    l2_mb = mem_hier.get("l2_cache_mb", hardware.get("l2_cache_mb", 40))

    try:
        lat = profile_latency(test_fn, inputs, warmup_iters=warmup,
                              timed_iters=timed, clear_l2=clear_l2,
                              l2_cache_mb=l2_mb)
        result.median_us = lat.median_us
        result.mean_us = lat.mean_us
        result.min_us = lat.min_us
        result.max_us = lat.max_us
        result.std_us = lat.std_us
    except torch.cuda.OutOfMemoryError as e:
        result.status = "oom"
        result.error_message = "OOM during profiling: %s" % str(e)[:100]
        logger.warning("  %s OOM during latency profiling", impl_name)
        return result
    except Exception as e:
        logger.warning("  %s latency profiling error: %s", impl_name, e)

    try:
        mem = profile_memory(test_fn, inputs)
        result.peak_allocated_mb = mem.peak_allocated_mb
        result.peak_reserved_mb = mem.peak_reserved_mb
        result.overhead_mb = mem.overhead_mb
    except Exception as e:
        logger.warning("  %s memory profiling error: %s", impl_name, e)

    # --- Derived metrics ---
    try:
        flops_fn, bytes_fn = resolve_metrics(kernel_dir, category)
        flops = flops_fn(config)
        total_bytes = bytes_fn(config, dtype_str)

        if result.median_us and result.median_us > 0:
            sol = compute_sol(result.median_us, flops, total_bytes,
                              hardware, dtype_str)
            result.achieved_tflops = sol.achieved_tflops
            result.sol_compute_pct = sol.sol_compute_pct
            result.achieved_bw_gb_s = sol.achieved_bw_gb_s
            result.sol_memory_pct = sol.sol_memory_pct
            result.bottleneck = sol.bottleneck
    except Exception as e:
        logger.debug("  %s metrics computation error: %s", impl_name, e)

    return result


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def _load_existing_results(results_dir: Path) -> Dict[str, Any]:
    """Load existing timing.json if it exists."""
    timing_path = results_dir / "timing.json"
    if timing_path.exists():
        with open(timing_path) as f:
            return json.load(f)
    return {}


def _result_exists(existing: dict, kernel: str, config_name: str,
                   impl: str, dtype: str) -> bool:
    """Check if a result already exists (for resumption)."""
    for r in existing.get("results", []):
        cfg = r.get("config", {})
        if cfg.get("name") == config_name and cfg.get("dtype") == dtype:
            if impl in r.get("implementations", {}):
                return True
    return False


def _save_results(kernel_name: str, results: List[EvalResult],
                  hardware: dict, results_dir: Path):
    """Write timing.json and append to summary.csv."""
    kernel_results_dir = results_dir / kernel_name
    kernel_results_dir.mkdir(parents=True, exist_ok=True)

    # Group results by config
    by_config = {}  # type: Dict[str, Dict[str, Any]]
    for r in results:
        key = "%s_%s" % (r.config_name, r.dtype)
        if key not in by_config:
            by_config[key] = {"config": r.config, "implementations": {}}
        impl_data = {k: v for k, v in asdict(r).items()
                     if k not in ("kernel", "config_name", "config",
                                  "implementation", "hardware", "timestamp")}
        by_config[key]["implementations"][r.implementation] = impl_data

    timing = {
        "kernel": kernel_name,
        "hardware": hardware.get("name", "unknown"),
        "cuda_version": getattr(torch.version, "cuda", "unknown"),
        "pytorch_version": torch.__version__,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "results": list(by_config.values()),
    }

    timing_path = kernel_results_dir / "timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2, default=str)

    # Append to summary CSV
    summary_path = results_dir / "summary.csv"
    write_header = not summary_path.exists()
    header = ("kernel,config_name,dtype,impl,correct,median_us,"
              "peak_allocated_mb,achieved_tflops,sol_compute_pct,"
              "achieved_bw_gb_s,sol_memory_pct,bottleneck,status")

    with open(summary_path, "a") as f:
        if write_header:
            f.write(header + "\n")
        for r in results:
            vals = [
                r.kernel, r.config_name, r.dtype, r.implementation,
                str(r.correct) if r.correct is not None else "",
                "%.1f" % r.median_us if r.median_us else "",
                "%.1f" % r.peak_allocated_mb if r.peak_allocated_mb else "",
                "%.3f" % r.achieved_tflops if r.achieved_tflops else "",
                "%.1f" % r.sol_compute_pct if r.sol_compute_pct else "",
                "%.1f" % r.achieved_bw_gb_s if r.achieved_bw_gb_s else "",
                "%.1f" % r.sol_memory_pct if r.sol_memory_pct else "",
                r.bottleneck or "",
                r.status,
            ]
            f.write(",".join(str(v) for v in vals) + "\n")


# ---------------------------------------------------------------------------
# Kernel listing
# ---------------------------------------------------------------------------

def list_kernels():
    """List all kernels and their implementation status."""
    print("%-40s  %-8s %-8s %-8s %-8s" % (
        "kernel", "ref", "triton", "tilelang", "category"))
    print("-" * 80)

    for kdir in sorted(KERNELS_DIR.iterdir()):
        if not kdir.is_dir():
            continue
        meta_path = kdir / "metadata.json"
        category = ""
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            category = meta.get("category", "")

        ref = "yes" if (kdir / "reference.py").exists() else "-"
        triton = "yes" if (kdir / "triton_impl.py").exists() else "-"
        tilelang = "yes" if (kdir / "tilelang_impl.py").exists() else "-"

        # Check NOT_IMPLEMENTED flags
        for impl_name, fname in [("triton", "triton_impl.py"),
                                 ("tilelang", "tilelang_impl.py")]:
            fpath = kdir / fname
            if fpath.exists():
                try:
                    mod = import_module_from_path(fpath)
                    if getattr(mod, "NOT_IMPLEMENTED", False):
                        if impl_name == "triton":
                            triton = "stub"
                        else:
                            tilelang = "stub"
                except Exception:
                    pass

        print("%-40s  %-8s %-8s %-8s %-8s" % (
            kdir.name, ref, triton, tilelang, category))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_kernel(
    kernel_name: str,
    hardware: dict,
    sweep_mode: str = "prioritized",
    correctness_only: bool = False,
    shape_filter: Optional[str] = None,
    dtype_filter: Optional[str] = None,
    extra_filters: Optional[Dict[str, str]] = None,
    force: bool = False,
):
    """Run evaluation for a single kernel."""
    kernel_dir = KERNELS_DIR / kernel_name
    if not kernel_dir.is_dir():
        print("Kernel not found: %s" % kernel_name)
        return []

    logger = setup_logger(kernel_name)
    logger.info("Evaluating kernel: %s", kernel_name)

    # Load configs
    input_cfg_path = kernel_dir / "input_config.json"
    meta_path = kernel_dir / "metadata.json"

    if not input_cfg_path.exists():
        logger.warning("No input_config.json for %s, skipping", kernel_name)
        return []

    with open(input_cfg_path) as f:
        input_config = json.load(f)
    category = ""
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        category = meta.get("category", "")

    # Generate sweep configs
    configs = _generate_sweep_configs(
        input_config, sweep_mode,
        shape_filter=shape_filter,
        dtype_filter=dtype_filter,
        extra_filters=extra_filters,
    )
    logger.info("  %d configs to evaluate", len(configs))

    # Discover implementations
    impls = _discover_implementations(kernel_dir)

    # Resolve input generator
    try:
        gen_fn = resolve_input_gen(kernel_dir, category)
    except ValueError as e:
        logger.error("  %s", e)
        return []

    # Load existing results for resumption
    existing = _load_existing_results(RESULTS_DIR / kernel_name) if not force else {}

    all_results = []

    for ci, config in enumerate(configs):
        config_name = config.get("name", "cfg_%d" % ci)
        dtype_str = config.get("dtype", "fp32")
        logger.info("  Config [%d/%d]: %s dtype=%s",
                     ci + 1, len(configs), config_name, dtype_str)

        # Generate inputs
        try:
            inputs = gen_fn(
                kernel_name=kernel_name,
                category=category,
                config=config,
                dtype=dtype_str,
            )
        except Exception as e:
            logger.error("    Input generation failed: %s", e)
            continue

        # Reference run — use reference_dtype (default fp32) for higher precision
        ref_path = kernel_dir / "reference.py"
        if not ref_path.exists():
            logger.error("    No reference.py found")
            continue

        correctness_cfg = input_config.get("correctness", {})
        ref_dtype_str = correctness_cfg.get("reference_dtype", "fp32")
        ref_dtype = get_torch_dtype(ref_dtype_str)

        try:
            ref_module = import_module_from_path(ref_path)
            ref_fn = ref_module.reference

            # Cast inputs to reference dtype for golden run
            ref_inputs = _deep_copy_inputs(inputs)
            for k, v in ref_inputs.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    ref_inputs[k] = v.to(ref_dtype)

            with torch.no_grad():
                golden_outputs = ref_fn(ref_inputs)
        except Exception as e:
            logger.error("    Reference run failed: %s: %s",
                         type(e).__name__, e)
            continue

        # Evaluate each implementation
        for impl_name, impl_path in sorted(impls.items()):
            # Skip if result already exists (resumption)
            if _result_exists(existing, kernel_name, config_name,
                              impl_name, dtype_str):
                logger.info("    %s: cached (use --force to re-run)", impl_name)
                continue

            try:
                fn, not_impl, module = _load_impl_callable(impl_name, impl_path)
            except Exception as e:
                r = EvalResult(
                    kernel=kernel_name, config_name=config_name,
                    config=config, implementation=impl_name,
                    status="import_error",
                    error_message="%s: %s" % (type(e).__name__, e),
                    hardware=hardware.get("name", ""),
                    dtype=dtype_str,
                    timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                )
                all_results.append(r)
                logger.warning("    %s import error: %s", impl_name, e)
                continue

            if not_impl:
                r = EvalResult(
                    kernel=kernel_name, config_name=config_name,
                    config=config, implementation=impl_name,
                    status="not_implemented",
                    hardware=hardware.get("name", ""),
                    dtype=dtype_str,
                    timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                )
                all_results.append(r)
                logger.info("    %s: not implemented", impl_name)
                continue

            r = _evaluate_single(
                kernel_name=kernel_name,
                config=config,
                impl_name=impl_name,
                fn=fn,
                golden_outputs=golden_outputs,
                inputs=_deep_copy_inputs(inputs),
                input_config=input_config,
                hardware=hardware,
                category=category,
                kernel_dir=kernel_dir,
                correctness_only=correctness_only,
                logger=logger,
            )
            all_results.append(r)

            status_str = "PASS" if r.correct else ("FAIL" if r.correct is False else r.status)
            extra = ""
            if r.median_us:
                extra = " %.1f us" % r.median_us
            if r.achieved_tflops:
                extra += " %.1f TFLOPS" % r.achieved_tflops
            logger.info("    %s: %s%s", impl_name, status_str, extra)

        # Cleanup between configs
        torch.cuda.empty_cache()

    # Save results
    if all_results:
        _save_results(kernel_name, all_results, hardware, RESULTS_DIR)
        logger.info("  Results saved to %s/%s/", RESULTS_DIR, kernel_name)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ViperBench evaluation runner")
    parser.add_argument("--kernel", type=str, help="Kernel name to evaluate")
    parser.add_argument("--all", action="store_true", help="Run all kernels")
    parser.add_argument("--sweep", choices=["prioritized", "full"],
                        default="prioritized", help="Sweep mode")
    parser.add_argument("--correctness-only", action="store_true",
                        help="Skip profiling, only check correctness")
    parser.add_argument("--shape", type=str, help="Filter to specific shape config")
    parser.add_argument("--dtype", type=str, help="Filter to specific dtype")
    parser.add_argument("--hardware", type=str,
                        help="Path to hardware config JSON (auto-detect if omitted)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results exist")
    parser.add_argument("--list", action="store_true",
                        help="List kernels and implementation status")
    args = parser.parse_args()

    if args.list:
        list_kernels()
        return

    if not args.kernel and not args.all:
        parser.error("Specify --kernel <name> or --all")

    # Load hardware config
    if args.hardware:
        hardware = load_hardware_config(Path(args.hardware))
    else:
        hardware = detect_hardware()

    print("Hardware: %s" % hardware.get("name", "unknown"))
    print("PyTorch %s, CUDA %s" % (torch.__version__, torch.version.cuda))

    # Determine kernels to run
    if args.all:
        kernels = sorted([d.name for d in KERNELS_DIR.iterdir() if d.is_dir()])
    else:
        kernels = [args.kernel]

    print("Evaluating %d kernel(s), sweep=%s" % (len(kernels), args.sweep))
    print("=" * 70)

    total_results = []
    for ki, kname in enumerate(kernels):
        print("[%d/%d] %s" % (ki + 1, len(kernels), kname))
        results = run_kernel(
            kernel_name=kname,
            hardware=hardware,
            sweep_mode=args.sweep,
            correctness_only=args.correctness_only,
            shape_filter=args.shape,
            dtype_filter=args.dtype,
            force=args.force,
        )
        total_results.extend(results)
        torch.cuda.empty_cache()
        torch._dynamo.reset()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    n_pass = sum(1 for r in total_results if r.correct is True)
    n_fail = sum(1 for r in total_results if r.correct is False)
    n_skip = sum(1 for r in total_results if r.status in ("not_implemented", "import_error"))
    n_err = sum(1 for r in total_results if r.status in ("runtime_error", "oom"))
    print("  %d evaluated: %d pass, %d fail, %d skipped, %d errors" %
          (len(total_results), n_pass, n_fail, n_skip, n_err))


if __name__ == "__main__":
    main()
