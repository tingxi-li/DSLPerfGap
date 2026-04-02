"""
Latency statistics, bandwidth computation, and IO byte models.
Based on EVALUATION_SPEC.md Sections 2.2, 3.3, and 7.
"""

import statistics
from typing import Optional

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def tensor_bytes(t: torch.Tensor) -> int:
    """Total bytes occupied by a tensor's elements."""
    return t.nelement() * t.element_size()


def total_tensor_bytes(tensors) -> int:
    """Sum of tensor_bytes for every torch.Tensor in *tensors* (skip non-tensors)."""
    return sum(tensor_bytes(t) for t in tensors if isinstance(t, torch.Tensor))


def model_param_bytes(model: torch.nn.Module) -> int:
    """Sum of bytes across all parameters of a module."""
    return sum(p.nelement() * p.element_size() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# Latency statistics  (Section 3.3)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_latency_stats(timings_ms: list[float]) -> dict:
    """Return latency statistics from *R* timing samples (milliseconds).

    Keys: latency_ms_mean, latency_ms_median, latency_ms_std,
          latency_ms_p95, latency_ms_min.
    All values are None when *timings_ms* is empty.
    """
    if not timings_ms:
        return {
            "latency_ms_mean": None,
            "latency_ms_median": None,
            "latency_ms_std": None,
            "latency_ms_p95": None,
            "latency_ms_min": None,
        }

    arr = np.asarray(timings_ms, dtype=np.float64)
    return {
        "latency_ms_mean": float(np.mean(arr)),
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_min": float(np.min(arr)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Bandwidth  (Section 2.2)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bandwidth_gbps(io_bytes: int, latency_seconds: float) -> float:
    """Effective bandwidth in GB/s.  Returns 0.0 when latency is zero."""
    if latency_seconds == 0.0:
        return 0.0
    return io_bytes / latency_seconds / 1e9


# ═══════════════════════════════════════════════════════════════════════════════
# IO byte models — per category  (Section 7)
# ═══════════════════════════════════════════════════════════════════════════════

def io_bytes_elementwise(inputs, outputs, model=None) -> int:
    """Sum of all input bytes + all output bytes."""
    return total_tensor_bytes(inputs) + total_tensor_bytes(outputs)


def io_bytes_matmul(inputs, outputs, model=None) -> int:
    """bytes(A) + bytes(B) + bytes(C).  Include model weight params if present."""
    total = total_tensor_bytes(inputs) + total_tensor_bytes(outputs)
    if model is not None:
        total += model_param_bytes(model)
    return total


def io_bytes_conv(inputs, outputs, model=None) -> int:
    """bytes(input) + bytes(output) + model parameter bytes (weights + bias)."""
    total = total_tensor_bytes(inputs) + total_tensor_bytes(outputs)
    if model is not None:
        total += model_param_bytes(model)
    return total


def io_bytes_reduction(inputs, outputs, model=None) -> int:
    """bytes(input) + bytes(output)."""
    return total_tensor_bytes(inputs) + total_tensor_bytes(outputs)


def io_bytes_attention(inputs, outputs, model=None) -> int:
    """bytes(Q) + bytes(K) + bytes(V) + bytes(output).

    Falls back to summing all inputs when fewer than 3 tensors are provided.
    """
    return total_tensor_bytes(inputs) + total_tensor_bytes(outputs)


def io_bytes_loss(inputs, outputs, model=None) -> int:
    """bytes(all inputs) + bytes(output)."""
    return total_tensor_bytes(inputs) + total_tensor_bytes(outputs)


def io_bytes_normalization(inputs, outputs, model=None) -> int:
    """bytes(input) + bytes(output) + model parameter bytes."""
    total = total_tensor_bytes(inputs) + total_tensor_bytes(outputs)
    if model is not None:
        total += model_param_bytes(model)
    return total


def io_bytes_pooling(inputs, outputs, model=None) -> int:
    """bytes(input) + bytes(output)."""
    return total_tensor_bytes(inputs) + total_tensor_bytes(outputs)


def io_bytes_model(inputs, outputs, model=None) -> int:
    """sum(all param bytes) + bytes(input) + bytes(output)."""
    total = total_tensor_bytes(inputs) + total_tensor_bytes(outputs)
    if model is not None:
        total += model_param_bytes(model)
    return total


def io_bytes_default(inputs, outputs, model=None) -> int:
    """Fallback: sum(input bytes) + sum(output bytes)."""
    return total_tensor_bytes(inputs) + total_tensor_bytes(outputs)


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch table  (Section 7)
# ═══════════════════════════════════════════════════════════════════════════════

IO_BYTES_DISPATCH: dict = {
    # matmul family
    "matmul":              io_bytes_matmul,
    "fused_gemm":          io_bytes_matmul,
    "fused_matmul":        io_bytes_matmul,
    "quantization":        io_bytes_matmul,
    # elementwise family
    "activation":          io_bytes_elementwise,
    "elementwise":         io_bytes_elementwise,
    "dropout":             io_bytes_elementwise,
    "cumulative":          io_bytes_elementwise,
    "embedding":           io_bytes_elementwise,
    # conv family
    "conv":                io_bytes_conv,
    "fused_conv":          io_bytes_conv,
    "fused_convtranspose": io_bytes_conv,
    # normalization
    "normalization":       io_bytes_normalization,
    # pooling
    "pooling":             io_bytes_pooling,
    # reduction
    "reduction":           io_bytes_reduction,
    # loss
    "loss":                io_bytes_loss,
    # attention
    "attention":           io_bytes_attention,
    # model family
    "model_cnn":           io_bytes_model,
    "model_transformer":   io_bytes_model,
    "model_rnn":           io_bytes_model,
    "model_other":         io_bytes_model,
    # catch-all
    "specialized":         io_bytes_default,
}


def compute_io_bytes(category: str, inputs, outputs, model=None) -> int:
    """Dispatch to the category-specific IO byte model.  Falls back to default."""
    fn = IO_BYTES_DISPATCH.get(category, io_bytes_default)
    return fn(inputs, outputs, model)
