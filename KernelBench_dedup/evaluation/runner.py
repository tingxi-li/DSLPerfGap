"""
Orchestration layer for the KernelBench evaluation system.

Integrates config, case_generator, correctness, metrics, and reporter
into a working evaluation pipeline.
"""

import gc
import importlib.util
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import (
    DEFAULT_DTYPES,
    DEFAULT_LAYOUTS,
    DEFAULT_SEED,
    DEFAULT_SHAPE_FAMILIES,
    DEFAULT_SIZE_BUCKETS,
    DEFAULT_VALUE_DISTS,
    FLOAT_DTYPES,
    TIMEOUT_SECONDS,
    TIMED_RUNS,
    WARMUP_RUNS,
    Layout,
    ShapeFamily,
    Status,
    TestCaseResult,
    TestCaseSpec,
    ValueDist,
)
from .case_generator import (
    generate_test_cases,
    get_kernel_category,
    load_categories,
    prepare_inputs_from_spec,
)
from .correctness import compute_correctness_metrics, flatten_outputs
from .metrics import (
    compute_bandwidth_gbps,
    compute_io_bytes,
    compute_latency_stats,
    total_tensor_bytes,
)
from .reporter import (
    print_console_summary,
    print_kernel_detail,
    save_results_json,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent
LEVELS = ["level1", "level2", "level3", "tritonbench"]


# ═══════════════════════════════════════════════════════════════════════════════
# Environment metadata (Section 8)
# ═══════════════════════════════════════════════════════════════════════════════


def collect_environment() -> dict:
    """Gather environment metadata for the evaluation report."""
    env: Dict[str, Any] = {}

    # GPU device
    try:
        env["device"] = torch.cuda.get_device_name()
    except Exception:
        env["device"] = "unknown"

    # Driver / CUDA version
    env["driver_version"] = getattr(torch.version, "cuda", "unknown")

    # PyTorch version
    env["pytorch_version"] = torch.__version__

    # OS
    env["os"] = platform.platform()

    # CPU model
    cpu_model = platform.processor()
    if not cpu_model:
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            cpu_model = "unknown"
    env["cpu_model"] = cpu_model

    # Seed and determinism
    env["seed"] = DEFAULT_SEED
    env["deterministic_mode"] = True

    # Git commit
    try:
        env["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(BASE_DIR),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        env["git_commit"] = "unknown"

    return env


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism
# ═══════════════════════════════════════════════════════════════════════════════


def set_deterministic(seed: int = DEFAULT_SEED):
    """Set all seeds and deterministic flags for reproducibility."""
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel discovery
# ═══════════════════════════════════════════════════════════════════════════════


def discover_kernels(
    levels: Optional[List[str]] = None,
    kernel_filter: Optional[str] = None,
) -> List[Tuple[str, str, Path]]:
    """Scan level directories for kernel directories containing pytorch_impl.py.

    Returns list of (level, name, kdir) tuples.
    """
    levels = levels or LEVELS
    kernels: List[Tuple[str, str, Path]] = []
    for level in levels:
        level_dir = BASE_DIR / level
        if not level_dir.is_dir():
            continue
        for kdir in sorted(level_dir.iterdir()):
            if not kdir.is_dir():
                continue
            impl = kdir / "pytorch_impl.py"
            if not impl.exists():
                continue
            name = kdir.name
            if kernel_filter and kernel_filter not in name:
                continue
            kernels.append((level, name, kdir))
    return kernels


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic import
# ═══════════════════════════════════════════════════════════════════════════════


def load_kernel_module(kdir: Path):
    """Import pytorch_impl.py from a kernel directory."""
    fpath = kdir / "pytorch_impl.py"
    kdir_str = str(kdir)
    if kdir_str not in sys.path:
        sys.path.insert(0, kdir_str)
    spec = importlib.util.spec_from_file_location(
        f"pytorch_impl_{kdir.name}", str(fpath)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmarking
# ═══════════════════════════════════════════════════════════════════════════════


def benchmark_kernel(
    model,
    inputs: list,
    warmup: int = WARMUP_RUNS,
    timed: int = TIMED_RUNS,
) -> List[float]:
    """Run warmup + timed iterations. Returns list of per-iteration latencies in ms."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(*inputs)
        torch.cuda.synchronize()

    # Timed runs
    timings: List[float] = []
    for _ in range(timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(*inputs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)

    return timings


# ═══════════════════════════════════════════════════════════════════════════════
# Timeout support
# ═══════════════════════════════════════════════════════════════════════════════


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Kernel execution timed out")


# ═══════════════════════════════════════════════════════════════════════════════
# Single test case execution
# ═══════════════════════════════════════════════════════════════════════════════


def _dtype_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to a short string like 'float32'."""
    return str(dtype).replace("torch.", "")


def _move_inputs_to_cuda(inputs: list) -> list:
    """Move a list of inputs to CUDA. Non-tensors pass through."""
    out = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            out.append(inp.cuda())
        else:
            out.append(inp)
    return out


def _cast_inputs(inputs: list, dtype: torch.dtype) -> list:
    """Cast float tensor inputs to the specified dtype. Non-float tensors pass through."""
    out = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor) and inp.is_floating_point():
            out.append(inp.to(dtype))
        else:
            out.append(inp)
    return out


def _get_shapes(inputs: list) -> list:
    """Extract shapes from inputs for reporting."""
    shapes = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            shapes.append(list(inp.shape))
    return shapes


def run_test_case(
    kernel_path: str,
    category: str,
    spec: TestCaseSpec,
    module,
    env: dict,
    timeout: int = TIMEOUT_SECONDS,
    warmup: int = WARMUP_RUNS,
    timed: int = TIMED_RUNS,
) -> TestCaseResult:
    """Execute a single test case for a kernel.

    Returns a TestCaseResult with status, correctness metrics, and latency stats.
    """
    # Base result with defaults
    result = TestCaseResult(
        op_name=kernel_path,
        backend="pytorch",
        reference_backend="pytorch_fp32",
        device=env.get("device", "unknown"),
        dtype_in=_dtype_str(spec.dtype),
        dtype_out=_dtype_str(spec.dtype),
        dtype_accum="float32",
        shape=[],
        layout=spec.layout.value,
        input_bytes_total=0,
        output_bytes_total=0,
        theoretical_io_bytes=0,
        warmup_runs=warmup,
        timed_runs=timed,
        status=Status.SKIPPED,
    )

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        set_deterministic()

        # Create model
        init_inputs = []
        if hasattr(module, "get_init_inputs"):
            init_inputs = module.get_init_inputs()
        model = module.Model(*init_inputs).cuda().eval()

        # Generate inputs from spec (size-bucket-aware shapes)
        spec_inputs = prepare_inputs_from_spec(spec, module, device="cuda")
        if spec_inputs is not None:
            ref_inputs = _cast_inputs(spec_inputs, torch.float32)
        else:
            # Fallback: use module's default inputs
            raw_inputs = module.get_inputs()
            ref_inputs = _move_inputs_to_cuda(raw_inputs)
            ref_inputs = _cast_inputs(ref_inputs, torch.float32)

        result.shape = _get_shapes(ref_inputs)
        result.input_bytes_total = sum(
            inp.nelement() * inp.element_size()
            for inp in ref_inputs
            if isinstance(inp, torch.Tensor)
        )

        # Run reference at fp32
        ref_model = model.float()
        with torch.no_grad():
            ref_output = ref_model(*ref_inputs)
        torch.cuda.synchronize()

        ref_output_tensors = flatten_outputs(ref_output)
        result.output_bytes_total = sum(
            t.nelement() * t.element_size() for t in ref_output_tensors
        )

        # Run test at specified dtype
        if spec.dtype == torch.float32:
            # For fp32, test output IS the reference (determinism check)
            test_output = ref_output
            test_inputs = ref_inputs
            test_model = ref_model
        else:
            # Cast model and inputs to test dtype
            try:
                test_model = model.to(spec.dtype)
            except Exception:
                test_model = model.half() if spec.dtype == torch.float16 else model
            test_inputs = _cast_inputs(ref_inputs, spec.dtype)
            with torch.no_grad():
                test_output = test_model(*test_inputs)
            torch.cuda.synchronize()

        # Correctness: compare test output (cast to fp32) vs reference
        test_output_for_cmp = test_output
        if spec.dtype != torch.float32:
            test_tensors = flatten_outputs(test_output)
            test_output_for_cmp = [t.float() for t in test_tensors]
            ref_output_for_cmp = flatten_outputs(ref_output)
        else:
            test_output_for_cmp = test_output
            ref_output_for_cmp = ref_output

        correctness = compute_correctness_metrics(
            test_output_for_cmp, ref_output_for_cmp, spec.dtype
        )

        result.max_abs_error = correctness["max_abs_error"]
        result.mean_abs_error = correctness["mean_abs_error"]
        result.max_rel_error = correctness["max_rel_error"]
        result.bad_elem_ratio = correctness["bad_elem_ratio"]
        result.correctness_pass = correctness["correctness_pass"]

        # Benchmark
        timings = benchmark_kernel(test_model, test_inputs, warmup=warmup, timed=timed)
        lat_stats = compute_latency_stats(timings)

        result.latency_ms_mean = lat_stats["latency_ms_mean"]
        result.latency_ms_median = lat_stats["latency_ms_median"]
        result.latency_ms_std = lat_stats["latency_ms_std"]
        result.latency_ms_p95 = lat_stats["latency_ms_p95"]
        result.latency_ms_min = lat_stats["latency_ms_min"]

        # IO and bandwidth
        test_inputs_list = [
            inp for inp in test_inputs if isinstance(inp, torch.Tensor)
        ]
        output_list = flatten_outputs(test_output)
        io_bytes = compute_io_bytes(category, test_inputs_list, output_list, test_model)
        result.theoretical_io_bytes = io_bytes

        if lat_stats["latency_ms_median"] and lat_stats["latency_ms_median"] > 0:
            result.bandwidth_gbps = compute_bandwidth_gbps(
                io_bytes, lat_stats["latency_ms_median"] / 1000.0
            )

        # Determine final status
        if correctness["correctness_pass"]:
            result.status = Status.PASS
        else:
            result.status = Status.FAIL_CORRECTNESS

    except _TimeoutError:
        result.status = Status.TIMEOUT
        result.notes = f"Exceeded {timeout}s timeout"
    except torch.cuda.OutOfMemoryError:
        result.status = Status.OOM
        result.notes = "CUDA out of memory"
        torch.cuda.empty_cache()
        gc.collect()
    except RuntimeError as e:
        err_lower = str(e).lower()
        if "unsupported" in err_lower or "not implemented" in err_lower:
            result.status = Status.NOT_SUPPORTED_BY_KERNEL
            result.notes = str(e)
        elif "out of memory" in err_lower:
            result.status = Status.OOM
            result.notes = str(e)
            torch.cuda.empty_cache()
            gc.collect()
        else:
            result.status = Status.FAIL_RUNTIME_ERROR
            result.notes = f"{type(e).__name__}: {e}"
    except Exception as e:
        result.status = Status.FAIL_RUNTIME_ERROR
        result.notes = f"{type(e).__name__}: {e}"
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Run all test cases for one kernel
# ═══════════════════════════════════════════════════════════════════════════════


def run_kernel_full(
    kernel_path: str,
    category: str,
    module,
    specs: List[TestCaseSpec],
    env: dict,
    timeout: int = TIMEOUT_SECONDS,
    warmup: int = WARMUP_RUNS,
    timed: int = TIMED_RUNS,
) -> List[TestCaseResult]:
    """Run all test cases for a single kernel. Clean GPU between cases."""
    results: List[TestCaseResult] = []
    for spec in specs:
        result = run_test_case(
            kernel_path, category, spec, module, env,
            timeout=timeout, warmup=warmup, timed=timed,
        )
        results.append(result)

        # Clean GPU between cases
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════


def run_all(
    levels: Optional[List[str]] = None,
    kernel_filter: Optional[str] = None,
    size_buckets: Optional[List[str]] = None,
    shape_families: Optional[List[ShapeFamily]] = None,
    dtypes: Optional[List[torch.dtype]] = None,
    layouts: Optional[List[Layout]] = None,
    value_dists: Optional[List[ValueDist]] = None,
    env: Optional[dict] = None,
    timeout: int = TIMEOUT_SECONDS,
    warmup: int = WARMUP_RUNS,
    timed: int = TIMED_RUNS,
    verbose: bool = False,
) -> List[TestCaseResult]:
    """Main loop: discover kernels, generate test cases, run them all.

    Returns aggregated list of TestCaseResult across all kernels.
    """
    if env is None:
        env = collect_environment()

    # Discover kernels
    kernels = discover_kernels(levels=levels, kernel_filter=kernel_filter)
    if not kernels:
        print("No kernels found.")
        return []

    # Load categories
    try:
        categories = load_categories()
    except Exception as e:
        print(f"Warning: Could not load categories.json: {e}")
        categories = {}

    total_kernels = len(kernels)
    all_results: List[TestCaseResult] = []

    for ki, (level, name, kdir) in enumerate(kernels, 1):
        kernel_path = f"{level}/{name}"
        category = get_kernel_category(kernel_path, categories)

        print(f"\n[{ki}/{total_kernels}] {kernel_path} (category={category})")

        # Load module
        try:
            module = load_kernel_module(kdir)
        except Exception as e:
            print(f"  IMPORT ERROR: {type(e).__name__}: {e}")
            # Record a single failure result for this kernel
            result = TestCaseResult(
                op_name=kernel_path,
                backend="pytorch",
                reference_backend="pytorch_fp32",
                device=env.get("device", "unknown"),
                dtype_in="float32",
                dtype_out="float32",
                dtype_accum="float32",
                shape=[],
                layout="contiguous",
                input_bytes_total=0,
                output_bytes_total=0,
                theoretical_io_bytes=0,
                warmup_runs=warmup,
                timed_runs=timed,
                status=Status.FAIL_RUNTIME_ERROR,
                notes=f"Import error: {type(e).__name__}: {e}",
            )
            all_results.append(result)
            continue

        # Generate test cases
        specs = generate_test_cases(
            kernel_path=kernel_path,
            category=category,
            module=module,
            size_buckets=size_buckets,
            shape_families=shape_families,
            dtypes=dtypes,
            layouts=layouts,
            value_dists=value_dists,
        )

        if not specs:
            print(f"  No test cases generated.")
            continue

        print(f"  {len(specs)} test cases")

        # Run all cases for this kernel
        kernel_results: List[TestCaseResult] = []
        for ci, spec in enumerate(specs, 1):
            spec_desc = (
                f"size={spec.size_bucket} shape={spec.shape_family.value} "
                f"dtype={_dtype_str(spec.dtype)} layout={spec.layout.value}"
            )
            print(f"  [{ci}/{len(specs)}] {spec_desc}", end="", flush=True)

            try:
                result = run_test_case(
                    kernel_path, category, spec, module, env,
                    timeout=timeout, warmup=warmup, timed=timed,
                )
            except Exception as e:
                # Catch-all so one case crash doesn't abort the kernel
                result = TestCaseResult(
                    op_name=kernel_path,
                    backend="pytorch",
                    reference_backend="pytorch_fp32",
                    device=env.get("device", "unknown"),
                    dtype_in=_dtype_str(spec.dtype),
                    dtype_out=_dtype_str(spec.dtype),
                    dtype_accum="float32",
                    shape=[],
                    layout=spec.layout.value,
                    input_bytes_total=0,
                    output_bytes_total=0,
                    theoretical_io_bytes=0,
                    warmup_runs=warmup,
                    timed_runs=timed,
                    status=Status.FAIL_RUNTIME_ERROR,
                    notes=f"Unhandled: {type(e).__name__}: {e}",
                )

            kernel_results.append(result)

            # Print inline status
            status_short = result.status.value
            lat_str = ""
            if result.latency_ms_median is not None:
                lat_str = f" {result.latency_ms_median:.3f}ms"
            print(f" -> {status_short}{lat_str}")

            # Clean GPU between cases
            torch.cuda.empty_cache()
            gc.collect()

        # Kernel summary line
        n_pass = sum(1 for r in kernel_results if r.status == Status.PASS)
        n_total = len(kernel_results)
        print(f"  Summary: {n_pass}/{n_total} PASS")

        if verbose:
            print_kernel_detail(kernel_path, kernel_results)

        all_results.extend(kernel_results)

        # Clean GPU between kernels
        torch.cuda.empty_cache()
        gc.collect()

    return all_results
