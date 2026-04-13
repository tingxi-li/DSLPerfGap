"""FLOP/byte counting and Speed-of-Light (SOL) computation for ViperBench."""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BYTES_PER_ELEMENT: Dict[str, int] = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp64": 8,
    "int8": 1,
}

# Per-element operation counts for common activations.
_ACTIVATION_OPS: Dict[str, int] = {
    "relu": 1,
    "gelu": 8,
    "sigmoid": 4,
    "tanh": 4,
    "selu": 2,
    "elu": 2,
    "leaky_relu": 1,
    "softmax": 5,
    "swish": 5,
}

# ---------------------------------------------------------------------------
# SOLResult
# ---------------------------------------------------------------------------


@dataclass
class SOLResult:
    """Results of a Speed-of-Light analysis for a single kernel run."""

    achieved_tflops: Optional[float] = None
    sol_compute_pct: Optional[float] = None
    achieved_bw_gb_s: Optional[float] = None
    sol_memory_pct: Optional[float] = None
    bottleneck: Optional[str] = None  # "compute" or "memory"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_batch(config: Dict[str, Any]) -> int:
    """Return the batch size from *config*, defaulting to 1."""
    return int(config.get("batch", config.get("B", 1)))


def _conv2d_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Compute the spatial output dimension for a Conv2d-style operation."""
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def _numel_from_config(config: Dict[str, Any]) -> Optional[int]:
    """Try to derive the total number of elements from common config keys."""
    if "numel" in config:
        return int(config["numel"])
    # Try (B, C, H, W)
    if all(k in config for k in ("B", "C", "H", "W")):
        return int(config["B"]) * int(config["C"]) * int(config["H"]) * int(config["W"])
    # Try (B, S, D) -- sequence-based
    if all(k in config for k in ("B", "S", "D")):
        return int(config["B"]) * int(config["S"]) * int(config["D"])
    # Try (M, N)
    if all(k in config for k in ("M", "N")):
        batch = _get_batch(config)
        return batch * int(config["M"]) * int(config["N"])
    # Try (N,) -- 1-D
    if "N" in config and "M" not in config:
        batch = _get_batch(config)
        return batch * int(config["N"])
    return None


# ---------------------------------------------------------------------------
# FLOP computation
# ---------------------------------------------------------------------------

def compute_flops(category: str, config: Dict[str, Any]) -> Optional[int]:
    """Estimate the number of floating-point operations for a kernel.

    Parameters
    ----------
    category:
        Kernel category string (e.g. ``"matmul"``, ``"conv"``, ``"attention"``).
    config:
        Dictionary of kernel parameters (shapes, strides, etc.).

    Returns
    -------
    int or None
        Estimated FLOPs, or ``None`` if the category is too complex to estimate.
    """
    cat = category.lower()

    # ------------------------------------------------------------------
    # Matrix multiplications (including fused variants)
    # ------------------------------------------------------------------
    if cat in ("matmul", "fused_matmul", "fused_gemm"):
        M = int(config.get("M", 0))
        N = int(config.get("N", 0))
        K = int(config.get("K", 0))
        if M == 0 or N == 0 or K == 0:
            return None
        batch = _get_batch(config)
        return 2 * M * N * K * batch

    # ------------------------------------------------------------------
    # Convolutions (including fused variants)
    # ------------------------------------------------------------------
    if cat in ("conv", "fused_conv", "fused_convtranspose"):
        B = int(config.get("B", config.get("batch", 1)))
        C_in = int(config.get("C_in", config.get("in_channels", 0)))
        C_out = int(config.get("C_out", config.get("out_channels", 0)))
        H = int(config.get("H", config.get("input_h", 0)))
        W = int(config.get("W", config.get("input_w", 0)))
        kH = int(config.get("kH", config.get("kernel_h", config.get("kernel_size", 0))))
        kW = int(config.get("kW", config.get("kernel_w", kH)))
        stride = int(config.get("stride", 1))
        padding = int(config.get("padding", 0))
        dilation = int(config.get("dilation", 1))
        groups = int(config.get("groups", 1))

        if C_in == 0 or C_out == 0 or H == 0 or W == 0 or kH == 0:
            return None

        H_out = _conv2d_output_size(H, kH, stride, padding, dilation)
        W_out = _conv2d_output_size(W, kW, stride, padding, dilation)

        return 2 * B * C_out * H_out * W_out * (C_in // groups) * kH * kW

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------
    if cat == "attention":
        B = int(config.get("B", config.get("batch", 1)))
        H = int(config.get("H", config.get("num_heads", 1)))
        D = int(config.get("D", config.get("head_dim", config.get("d_model", 0))))
        seq_q = int(config.get("seq_q", config.get("S", 0)))
        seq_kv = int(config.get("seq_kv", seq_q))

        if D == 0 or seq_q == 0:
            return None
        # QK^T: 2*B*H*S_q*S_kv*D  +  softmax: ~5*B*H*S_q*S_kv
        # AV:   2*B*H*S_q*D*S_kv
        # Simplified: 4 * B * H * S_q * S_kv * D
        return 4 * B * H * seq_q * seq_kv * D

    # ------------------------------------------------------------------
    # Normalization (layernorm, batchnorm, etc.)
    # ------------------------------------------------------------------
    if cat == "normalization":
        numel = _numel_from_config(config)
        if numel is None:
            return None
        return 5 * numel  # mean, var, normalize, scale, shift

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------
    if cat == "activation":
        numel = _numel_from_config(config)
        if numel is None:
            return None
        # Try to detect the specific activation from config.
        act_name = str(config.get("activation", config.get("act", ""))).lower()
        ops = _ACTIVATION_OPS.get(act_name, 1)
        return numel * ops

    # ------------------------------------------------------------------
    # Reduction (sum, mean, max, min, argmax, etc.)
    # ------------------------------------------------------------------
    if cat == "reduction":
        numel = _numel_from_config(config)
        if numel is None:
            return None
        return numel

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    if cat == "loss":
        numel = _numel_from_config(config)
        if numel is None:
            return None
        # Element-wise comparison + final reduction.
        reduction_numel = int(config.get("reduction_numel", _get_batch(config)))
        return numel * 2 + reduction_numel

    # ------------------------------------------------------------------
    # Cumulative ops (cumsum, cumprod)
    # ------------------------------------------------------------------
    if cat == "cumulative":
        numel = _numel_from_config(config)
        if numel is None:
            return None
        return numel

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------
    if cat == "pooling":
        B = int(config.get("B", config.get("batch", 1)))
        C = int(config.get("C", config.get("channels", 0)))
        H = int(config.get("H", config.get("input_h", 0)))
        W = int(config.get("W", config.get("input_w", 0)))
        kH = int(config.get("kH", config.get("kernel_h", config.get("kernel_size", 0))))
        kW = int(config.get("kW", config.get("kernel_w", kH)))
        stride = int(config.get("stride", kH))
        padding = int(config.get("padding", 0))

        if C == 0 or H == 0 or W == 0 or kH == 0:
            return None

        H_out = _conv2d_output_size(H, kH, stride, padding)
        W_out = _conv2d_output_size(W, kW, stride, padding)

        return B * C * H_out * W_out * kH * kW

    # ------------------------------------------------------------------
    # Embedding (table lookup)
    # ------------------------------------------------------------------
    if cat == "embedding":
        B = int(config.get("B", config.get("batch", 1)))
        S = int(config.get("S", config.get("seq_len", 0)))
        D = int(config.get("D", config.get("embed_dim", 0)))
        if S == 0 or D == 0:
            return None
        return B * S * D

    # ------------------------------------------------------------------
    # Elementwise / dropout -- simple element counts
    # ------------------------------------------------------------------
    if cat in ("elementwise", "dropout"):
        numel = _numel_from_config(config)
        if numel is None:
            return None
        return numel

    # ------------------------------------------------------------------
    # Categories that are too complex or not meaningful to estimate
    # ------------------------------------------------------------------
    if cat in ("quantization", "specialized") or cat.startswith("model_"):
        return None

    # Unknown category -- return None rather than guessing.
    return None


# ---------------------------------------------------------------------------
# Byte-transfer computation
# ---------------------------------------------------------------------------

def compute_bytes(category: str, config: Dict[str, Any], dtype: str) -> Optional[int]:
    """Estimate the number of bytes transferred (read + write) for a kernel.

    Parameters
    ----------
    category:
        Kernel category string.
    config:
        Dictionary of kernel parameters.
    dtype:
        Data type string (e.g. ``"fp16"``, ``"fp32"``).

    Returns
    -------
    int or None
        Estimated byte count, or ``None`` if it cannot be determined.
    """
    bpe = BYTES_PER_ELEMENT.get(dtype)
    if bpe is None:
        return None

    cat = category.lower()

    # ------------------------------------------------------------------
    # Matmul / fused matmul
    # ------------------------------------------------------------------
    if cat in ("matmul", "fused_matmul", "fused_gemm"):
        M = int(config.get("M", 0))
        N = int(config.get("N", 0))
        K = int(config.get("K", 0))
        if M == 0 or N == 0 or K == 0:
            return None
        batch = _get_batch(config)
        return (M * K + K * N + M * N) * bpe * batch

    # ------------------------------------------------------------------
    # Activation / elementwise / dropout (read + write entire tensor)
    # ------------------------------------------------------------------
    if cat in ("activation", "elementwise", "dropout"):
        numel = _numel_from_config(config)
        if numel is None:
            return None
        return 2 * numel * bpe

    # ------------------------------------------------------------------
    # Normalization (read input + params, write output)
    # ------------------------------------------------------------------
    if cat == "normalization":
        numel = _numel_from_config(config)
        if numel is None:
            return None
        return 3 * numel * bpe

    # ------------------------------------------------------------------
    # Reduction (read full, write reduced)
    # ------------------------------------------------------------------
    if cat == "reduction":
        numel = _numel_from_config(config)
        if numel is None:
            return None
        output_numel = int(config.get("output_numel", _get_batch(config)))
        return (numel + output_numel) * bpe

    # ------------------------------------------------------------------
    # Attention (read Q, K, V; write O)
    # ------------------------------------------------------------------
    if cat == "attention":
        B = int(config.get("B", config.get("batch", 1)))
        H = int(config.get("H", config.get("num_heads", 1)))
        S = int(config.get("S", 0))
        seq_q = int(config.get("seq_q", S))
        seq_kv = int(config.get("seq_kv", seq_q))
        kv_heads = int(config.get("kv_heads", H))
        D = int(config.get("D", config.get("head_dim", config.get("d_model", 0))))
        if seq_q == 0 or D == 0:
            return None
        # Q: (B, H, seq_q, D), K: (B, kv_heads, seq_kv, D), V: same, O: (B, H, seq_q, D)
        q_bytes = B * H * seq_q * D * bpe
        kv_bytes = 2 * B * kv_heads * seq_kv * D * bpe
        o_bytes = B * H * seq_q * D * bpe
        return q_bytes + kv_bytes + o_bytes

    # ------------------------------------------------------------------
    # Loss (read predictions + targets, write scalar)
    # ------------------------------------------------------------------
    if cat == "loss":
        numel = _numel_from_config(config)
        if numel is None:
            return None
        # Read pred + target, write a scalar output.
        return 2 * numel * bpe + bpe

    # ------------------------------------------------------------------
    # Convolution (input + weight + output tensors)
    # ------------------------------------------------------------------
    if cat in ("conv", "fused_conv", "fused_convtranspose"):
        B = int(config.get("B", config.get("batch", 1)))
        C_in = int(config.get("C_in", config.get("in_channels", 0)))
        C_out = int(config.get("C_out", config.get("out_channels", 0)))
        H = int(config.get("H", config.get("input_h", 0)))
        W = int(config.get("W", config.get("input_w", 0)))
        kH = int(config.get("kH", config.get("kernel_h", config.get("kernel_size", 0))))
        kW = int(config.get("kW", config.get("kernel_w", kH)))
        stride = int(config.get("stride", 1))
        padding = int(config.get("padding", 0))
        dilation = int(config.get("dilation", 1))
        groups = int(config.get("groups", 1))

        if C_in == 0 or C_out == 0 or H == 0 or W == 0 or kH == 0:
            return None

        H_out = _conv2d_output_size(H, kH, stride, padding, dilation)
        W_out = _conv2d_output_size(W, kW, stride, padding, dilation)

        input_numel = B * C_in * H * W
        weight_numel = C_out * (C_in // groups) * kH * kW
        output_numel = B * C_out * H_out * W_out

        return (input_numel + weight_numel + output_numel) * bpe

    # ------------------------------------------------------------------
    # Cumulative (read + write)
    # ------------------------------------------------------------------
    if cat == "cumulative":
        numel = _numel_from_config(config)
        if numel is None:
            return None
        return 2 * numel * bpe

    # ------------------------------------------------------------------
    # Pooling (input + output)
    # ------------------------------------------------------------------
    if cat == "pooling":
        B = int(config.get("B", config.get("batch", 1)))
        C = int(config.get("C", config.get("channels", 0)))
        H = int(config.get("H", config.get("input_h", 0)))
        W = int(config.get("W", config.get("input_w", 0)))
        kH = int(config.get("kH", config.get("kernel_h", config.get("kernel_size", 0))))
        kW = int(config.get("kW", config.get("kernel_w", kH)))
        stride = int(config.get("stride", kH))
        padding = int(config.get("padding", 0))

        if C == 0 or H == 0 or W == 0 or kH == 0:
            return None

        H_out = _conv2d_output_size(H, kH, stride, padding)
        W_out = _conv2d_output_size(W, kW, stride, padding)

        input_numel = B * C * H * W
        output_numel = B * C * H_out * W_out
        return (input_numel + output_numel) * bpe

    # ------------------------------------------------------------------
    # Embedding (read embedding rows, write output)
    # ------------------------------------------------------------------
    if cat == "embedding":
        B = int(config.get("B", config.get("batch", 1)))
        S = int(config.get("S", config.get("seq_len", 0)))
        D = int(config.get("D", config.get("embed_dim", 0)))
        if S == 0 or D == 0:
            return None
        # Read the looked-up rows and write the output tensor.
        return 2 * B * S * D * bpe

    # ------------------------------------------------------------------
    # Unsupported categories
    # ------------------------------------------------------------------
    return None


# ---------------------------------------------------------------------------
# Arithmetic intensity
# ---------------------------------------------------------------------------

def compute_arithmetic_intensity(
    category: str,
    config: Dict[str, Any],
    dtype: str,
) -> Optional[float]:
    """Compute the arithmetic intensity (FLOPs / bytes) for a kernel.

    Returns
    -------
    float or None
        Arithmetic intensity in FLOPs per byte, or ``None`` if either
        FLOPs or bytes cannot be estimated.
    """
    flops = compute_flops(category, config)
    nbytes = compute_bytes(category, config, dtype)
    if flops is None or nbytes is None or nbytes == 0:
        return None
    return flops / nbytes


# ---------------------------------------------------------------------------
# Peak-throughput helpers
# ---------------------------------------------------------------------------

def _resolve_peak_tflops(hardware: Dict[str, Any], dtype: str) -> Optional[float]:
    """Extract the peak TFLOPS for *dtype* from a hardware config dict.

    Supports both the nested layout (``compute_characteristics.peak_*``) and
    a flat layout where the peak fields live at the top level.
    """
    # Map dtype strings to the suffix used in hardware JSON keys.
    _DTYPE_TO_KEY: Dict[str, str] = {
        "fp16": "peak_fp16_tensor_tflops",
        "bf16": "peak_bf16_tensor_tflops",
        "fp32": "peak_fp32_tflops",
        "fp64": "peak_fp64_tflops",
        "tf32": "peak_tf32_tensor_tflops",
        "int8": "peak_int8_tops",
    }
    key = _DTYPE_TO_KEY.get(dtype)
    if key is None:
        return None

    # Try nested structure first.
    cc = hardware.get("compute_characteristics")
    if isinstance(cc, dict):
        val = cc.get(key)
        if val is not None:
            return float(val)

    # Try flat structure.
    val = hardware.get(key)
    if val is not None:
        return float(val)

    return None


def _resolve_peak_bw(hardware: Dict[str, Any]) -> Optional[float]:
    """Extract peak global memory bandwidth (GB/s) from a hardware config."""
    mh = hardware.get("memory_hierarchy")
    if isinstance(mh, dict):
        val = mh.get("global_memory_bandwidth_gb_s")
        if val is not None:
            return float(val)

    # Flat fallback.
    val = hardware.get("memory_bandwidth_gb_s")
    if val is not None:
        return float(val)

    return None


# ---------------------------------------------------------------------------
# SOL computation
# ---------------------------------------------------------------------------

def compute_sol(
    latency_us: float,
    flops: Optional[int],
    bytes_transferred: Optional[int],
    hardware: Dict[str, Any],
    dtype: str,
) -> SOLResult:
    """Compute the Speed-of-Light metrics for a kernel execution.

    Parameters
    ----------
    latency_us:
        Measured kernel latency in microseconds.
    flops:
        Estimated FLOPs for the kernel (may be ``None``).
    bytes_transferred:
        Estimated bytes transferred (may be ``None``).
    hardware:
        Hardware specification dictionary (see ``configs/hardware/*.json``).
    dtype:
        Data type string used by the kernel (e.g. ``"fp16"``).

    Returns
    -------
    SOLResult
        Populated Speed-of-Light result.
    """
    result = SOLResult()

    if latency_us <= 0:
        return result

    latency_s = latency_us * 1e-6

    # ---- Compute SOL ----
    peak_tflops = _resolve_peak_tflops(hardware, dtype)

    if flops is not None:
        result.achieved_tflops = flops / latency_s / 1e12
        if peak_tflops is not None and peak_tflops > 0:
            result.sol_compute_pct = result.achieved_tflops / peak_tflops * 100

    # ---- Memory SOL ----
    peak_bw = _resolve_peak_bw(hardware)

    if bytes_transferred is not None:
        result.achieved_bw_gb_s = bytes_transferred / latency_s / 1e9
        if peak_bw is not None and peak_bw > 0:
            result.sol_memory_pct = result.achieved_bw_gb_s / peak_bw * 100

    # ---- Bottleneck classification via roofline ----
    if (
        flops is not None
        and bytes_transferred is not None
        and bytes_transferred > 0
        and peak_tflops is not None
        and peak_tflops > 0
        and peak_bw is not None
        and peak_bw > 0
    ):
        arithmetic_intensity = flops / bytes_transferred  # FLOP / byte
        # Ridge point: peak_compute (FLOP/s) / peak_bw (byte/s)
        ridge_point = (peak_tflops * 1e12) / (peak_bw * 1e9)
        result.bottleneck = "compute" if arithmetic_intensity > ridge_point else "memory"

    return result


# ---------------------------------------------------------------------------
# Per-kernel metric overrides
# ---------------------------------------------------------------------------

def resolve_metrics(
    kernel_dir: Path,
    category: str,
) -> Tuple[Callable[..., Optional[int]], Callable[..., Optional[int]]]:
    """Return ``(compute_flops_fn, compute_bytes_fn)`` for a kernel.

    If ``kernel_dir/metrics.py`` exists it is dynamically imported and its
    ``compute_flops`` / ``compute_bytes`` callables are used.  Otherwise the
    central :func:`compute_flops` and :func:`compute_bytes` functions from
    this module are returned, pre-bound to *category*.

    Parameters
    ----------
    kernel_dir:
        Path to the kernel directory (e.g.
        ``ViperBench/level1/matmul_basic/``).
    category:
        Kernel category string to bind when using the central functions.

    Returns
    -------
    tuple[Callable, Callable]
        A pair of ``(flops_fn, bytes_fn)`` where each callable has the
        signature ``(config, [dtype]) -> Optional[int]``.
    """
    override_path = Path(kernel_dir) / "metrics.py"

    if override_path.is_file():
        module_name = f"_metrics_override_{override_path.parent.name}"
        spec = importlib.util.spec_from_file_location(module_name, str(override_path))
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)

            flops_fn = getattr(mod, "compute_flops", None)
            bytes_fn = getattr(mod, "compute_bytes", None)

            if flops_fn is not None and bytes_fn is not None:
                return flops_fn, bytes_fn

    # Fall back to the central implementations, binding the category.
    def _bound_flops(config: Dict[str, Any]) -> Optional[int]:
        return compute_flops(category, config)

    def _bound_bytes(config: Dict[str, Any], dtype: str = "fp32") -> Optional[int]:
        return compute_bytes(category, config, dtype)

    return _bound_flops, _bound_bytes
