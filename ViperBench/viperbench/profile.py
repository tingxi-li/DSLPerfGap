"""Latency and memory profiling for ViperBench kernels."""
from __future__ import annotations

import statistics
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import torch


@dataclass
class LatencyResult:
    """Aggregated latency statistics from timed kernel runs."""

    median_us: float
    mean_us: float
    min_us: float
    max_us: float
    std_us: float
    all_times_us: List[float] = field(default_factory=list)


@dataclass
class MemoryResult:
    """Peak memory consumption from a single kernel invocation."""

    peak_allocated_mb: float
    peak_reserved_mb: float
    input_memory_mb: float
    overhead_mb: float


def profile_latency(
    fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    inputs: Dict[str, Any],
    warmup_iters: int = 10,
    timed_iters: int = 100,
    clear_l2: bool = True,
    l2_cache_mb: int = 40,
) -> LatencyResult:
    """Profile GPU kernel latency using CUDA events.

    Parameters
    ----------
    fn : callable
        Kernel function with signature ``fn(inputs: dict) -> dict``.
    inputs : dict
        String-keyed mapping whose values are :class:`torch.Tensor` or
        non-tensor parameters (ints, bools, etc.).
    warmup_iters : int
        Number of untimed warm-up invocations.
    timed_iters : int
        Number of timed invocations.
    clear_l2 : bool
        If *True*, flush the GPU L2 cache between each timed iteration by
        allocating and writing a scratch buffer.
    l2_cache_mb : int
        Size (in MiB) of the scratch buffer used to flush L2.

    Returns
    -------
    LatencyResult
    """
    # --- warmup -----------------------------------------------------------
    for _ in range(warmup_iters):
        fn(inputs)

    # Pre-compute scratch-buffer size (bytes).  We want at least
    # ``l2_cache_mb`` MiB of int8 elements.
    l2_scratch_elems = l2_cache_mb * 1024 * 1024  # 1 byte per int8

    # --- timed iterations -------------------------------------------------
    times_us: List[float] = []
    for _ in range(timed_iters):
        # Optionally flush the L2 cache.
        if clear_l2:
            scratch = torch.empty(l2_scratch_elems, dtype=torch.int8, device="cuda")
            scratch.zero_()
            del scratch

        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        fn(inputs)
        end_event.record()

        torch.cuda.synchronize()

        elapsed_ms: float = start_event.elapsed_time(end_event)
        times_us.append(elapsed_ms * 1000.0)

    # --- aggregate --------------------------------------------------------
    times_us.sort()
    median_us = float(statistics.median(times_us))
    mean_us = float(statistics.mean(times_us))
    min_us = float(min(times_us))
    max_us = float(max(times_us))
    std_us = float(statistics.stdev(times_us)) if len(times_us) > 1 else 0.0

    return LatencyResult(
        median_us=median_us,
        mean_us=mean_us,
        min_us=min_us,
        max_us=max_us,
        std_us=std_us,
        all_times_us=times_us,
    )


def profile_memory(
    fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    inputs: Dict[str, Any],
) -> MemoryResult:
    """Profile peak GPU memory usage of a single kernel invocation.

    Parameters
    ----------
    fn : callable
        Kernel function with signature ``fn(inputs: dict) -> dict``.
    inputs : dict
        String-keyed mapping whose values are :class:`torch.Tensor` or
        non-tensor parameters.

    Returns
    -------
    MemoryResult
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Compute memory occupied by input tensors.
    input_bytes = 0
    for value in inputs.values():
        if isinstance(value, torch.Tensor):
            input_bytes += value.element_size() * value.nelement()
    input_memory_mb = input_bytes / 1e6

    fn(inputs)
    torch.cuda.synchronize()

    peak_allocated_mb = torch.cuda.max_memory_allocated() / 1e6
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1e6
    overhead_mb = peak_allocated_mb - input_memory_mb

    return MemoryResult(
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,
        input_memory_mb=input_memory_mb,
        overhead_mb=overhead_mb,
    )


def check_gpu_clocks() -> bool:
    """Best-effort check whether GPU clocks are locked.

    Parses ``nvidia-smi -q -d CLOCK`` output looking for an indication that
    the applications-clock or max-clock values are pinned.  If clocks do not
    appear to be locked the function prints a helpful warning with the
    ``nvidia-smi`` command to lock them.

    Returns
    -------
    bool
        *True* if clocks appear to be locked (or the check could not be
        performed).  *False* if clocks are clearly *not* locked.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-d", "CLOCK"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # nvidia-smi unavailable or timed out — nothing we can do.
        return True

    if result.returncode != 0:
        return True

    output = result.stdout

    # We look for the "Applications Clocks" and "Max Clocks" sections.
    # If the applications-clock graphics frequency matches the max-clock
    # graphics frequency the clocks are effectively locked.
    apps_graphics: str | None = None
    max_graphics: str | None = None

    section = ""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Applications Clocks"):
            section = "apps"
        elif stripped.startswith("Max Clocks"):
            section = "max"
        elif stripped.startswith("Default Applications Clocks"):
            section = ""

        if section == "apps" and stripped.startswith("Graphics"):
            apps_graphics = stripped.split(":")[-1].strip()
        if section == "max" and stripped.startswith("Graphics"):
            max_graphics = stripped.split(":")[-1].strip()

    if apps_graphics is not None and max_graphics is not None:
        if apps_graphics == max_graphics:
            return True

        print(
            "WARNING: GPU clocks do not appear to be locked. "
            "Benchmark results may be noisy.\n"
            "Lock clocks with:\n"
            f"  sudo nvidia-smi -lgc {max_graphics.rstrip(' MHz').strip()},{max_graphics.rstrip(' MHz').strip()}\n"
            "Unlock later with:\n"
            "  sudo nvidia-smi -rgc"
        )
        return False

    # Could not determine — assume OK.
    return True
