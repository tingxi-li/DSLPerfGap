"""Central input generation module for ViperBench kernel benchmarks."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from .utils import DTYPE_MAP

# ---------------------------------------------------------------------------
# Category-specific generators
# ---------------------------------------------------------------------------
# Each generator has the signature:
#   (config: dict, dtype_str: str, device: str) -> dict[str, torch.Tensor | int | ...]
# ---------------------------------------------------------------------------


def _get_dtype(dtype_str: str) -> torch.dtype:
    """Resolve a short dtype string to a ``torch.dtype``."""
    return DTYPE_MAP.get(dtype_str, torch.float32)


# -- matmul ----------------------------------------------------------------

def _gen_matmul(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    M: int = config["M"]
    N: int = config["N"]
    K: int = config["K"]
    batch: int = config.get("batch", 1)
    transpose: str = config.get("transpose", "NN")
    structure: str = config.get("structure", "dense")

    # Determine raw 2-D shapes before transpose encoding
    if transpose == "NN":
        shape_a = (M, K)
        shape_b = (K, N)
    elif transpose == "NT":
        shape_a = (M, K)
        shape_b = (N, K)
    elif transpose == "TN":
        shape_a = (K, M)
        shape_b = (K, N)
    elif transpose == "TT":
        shape_a = (K, M)
        shape_b = (N, K)
    else:
        raise ValueError(f"Unknown transpose mode: {transpose}")

    def _make(shape: tuple) -> torch.Tensor:
        if batch > 1:
            return torch.randn(batch, *shape, dtype=dt, device=device)
        return torch.randn(*shape, dtype=dt, device=device)

    A = _make(shape_a)
    B = _make(shape_b)

    # Apply structure modifiers to A (shape-preserving)
    if structure == "dense":
        pass
    elif structure == "diagonal":
        # Zero out off-diagonal elements to create a rectangular diagonal mask
        # Works for any (rows, cols) shape — preserves original dimensions
        rows, cols = shape_a[-2], shape_a[-1]
        mask = torch.zeros(rows, cols, dtype=dt, device=device)
        diag_len = min(rows, cols)
        diag_vals = torch.randn(diag_len, dtype=dt, device=device)
        mask[:diag_len, :diag_len] = torch.diag(diag_vals)
        if batch > 1:
            A = mask.unsqueeze(0).expand(batch, -1, -1).clone()
        else:
            A = mask
    elif structure == "upper_triangular":
        A = torch.triu(A)
    elif structure == "lower_triangular":
        A = torch.tril(A)
    elif structure == "symmetric":
        # Symmetric requires square matrix (M == K for NN/NT transpose)
        rows, cols = shape_a[-2], shape_a[-1]
        if rows != cols:
            raise ValueError(
                "symmetric structure requires square A (%d != %d); "
                "skip this config" % (rows, cols)
            )
        A = (A + A.transpose(-1, -2)) / 2.0
    else:
        pass  # default to dense

    return {"A": A, "B": B}


# -- conv ------------------------------------------------------------------

def _gen_conv(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    B: int = config["B"]
    C_in: int = config["C_in"]
    C_out: int = config["C_out"]
    groups: int = config.get("groups", 1)
    stride: int = config.get("stride", 1)
    padding: int = config.get("padding", 0)
    dilation: int = config.get("dilation", 1)

    # 1-D convolution (has L and K but no H/W)
    if "L" in config and "H" not in config:
        L: int = config["L"]
        K_1d: int = config.get("K", config.get("kH", 3))
        inp = torch.randn(B, C_in, L, dtype=dt, device=device)
        weight = torch.randn(C_out, C_in // groups, K_1d, dtype=dt, device=device)
        bias = torch.randn(C_out, dtype=dt, device=device)
        return {
            "input": inp,
            "weight": weight,
            "bias": bias,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }

    # 3-D convolution (has D, H, W)
    if "D" in config and "H" in config and "W" in config:
        D: int = config["D"]
        H_3d: int = config["H"]
        W_3d: int = config["W"]
        kH: int = config.get("kH", 3)
        kW: int = config.get("kW", 3)
        kD: int = config.get("kD", kH)
        inp = torch.randn(B, C_in, D, H_3d, W_3d, dtype=dt, device=device)
        weight = torch.randn(
            C_out, C_in // groups, kD, kH, kW, dtype=dt, device=device
        )
        bias = torch.randn(C_out, dtype=dt, device=device)
        return {
            "input": inp,
            "weight": weight,
            "bias": bias,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }

    # 2-D convolution (default)
    H: int = config["H"]
    W: int = config["W"]
    kH = config.get("kH", 3)
    kW = config.get("kW", 3)

    inp = torch.randn(B, C_in, H, W, dtype=dt, device=device)
    weight = torch.randn(C_out, C_in // groups, kH, kW, dtype=dt, device=device)
    bias = torch.randn(C_out, dtype=dt, device=device)

    return {
        "input": inp,
        "weight": weight,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }


# -- attention -------------------------------------------------------------

def _gen_attention(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    B: int = config["B"]
    H: int = config["H"]
    D: int = config["D"]
    seq_q: int = config.get("seq_q", config.get("S", 128))
    seq_kv: int = config.get("seq_kv", seq_q)
    kv_heads: int = config.get("kv_heads", H)
    mask_mode: str = config.get("mask", "none")

    Q = torch.randn(B, H, seq_q, D, dtype=dt, device=device)
    K = torch.randn(B, kv_heads, seq_kv, D, dtype=dt, device=device)
    V = torch.randn(B, kv_heads, seq_kv, D, dtype=dt, device=device)

    out: Dict[str, Any] = {"Q": Q, "K": K, "V": V}

    if mask_mode == "causal":
        mask = torch.tril(
            torch.ones(seq_q, seq_kv, dtype=torch.bool, device=device)
        )
        out["mask"] = mask
    elif mask_mode.startswith("sparse_"):
        # "sparse_X" where X is a sparsity percentage (e.g. sparse_50)
        try:
            sparsity = int(mask_mode.split("_", 1)[1]) / 100.0
        except (ValueError, IndexError):
            sparsity = 0.5
        mask = torch.rand(seq_q, seq_kv, device=device) >= sparsity
        out["mask"] = mask
    # mask_mode == "none" -> no mask key

    return out


# -- normalization ---------------------------------------------------------

def _gen_normalization(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    B: int = config["B"]
    dims: List[int] = config["dims"]
    norm_type: str = config.get("norm_type", "layer_norm")

    inp = torch.randn(B, *dims, dtype=dt, device=device)

    if norm_type in ("batch_norm", "instance_norm"):
        normalized_shape = dims[0]  # channels dimension
    else:
        # layer_norm, rms_norm
        normalized_shape = dims[-1]

    weight = torch.randn(normalized_shape, dtype=dt, device=device)
    bias = torch.randn(normalized_shape, dtype=dt, device=device)

    out: Dict[str, Any] = {"input": inp, "weight": weight, "bias": bias}

    if norm_type == "group_norm":
        out["num_groups"] = config["num_groups"]

    return out


# -- activation / elementwise / cumulative / dropout -----------------------

def _gen_activation(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    dims: List[int] = config["dims"]
    return {"input": torch.randn(*dims, dtype=dt, device=device)}


_gen_elementwise = _gen_activation
_gen_cumulative = _gen_activation


# -- reduction -------------------------------------------------------------

def _gen_reduction(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    dims: List[int] = config["dims"]
    return {"input": torch.randn(*dims, dtype=dt, device=device)}


# -- loss ------------------------------------------------------------------

def _gen_loss(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    B: int = config["B"]
    C: int = config["C"]
    S: Optional[int] = config.get("S", None)

    if S is not None:
        prediction = torch.randn(B, S, C, dtype=dt, device=device)
        target = torch.randint(0, C, (B, S), device=device)
    else:
        prediction = torch.randn(B, C, dtype=dt, device=device)
        target = torch.randint(0, C, (B,), device=device)

    return {"prediction": prediction, "target": target}


# -- pooling ---------------------------------------------------------------

def _gen_pooling(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    B: int = config["B"]
    C: int = config["C"]
    H: int = config["H"]
    W: int = config["W"]
    kernel_size: int = config.get("kernel_size", 2)
    stride: int = config.get("stride", kernel_size)

    inp = torch.randn(B, C, H, W, dtype=dt, device=device)
    return {"input": inp, "kernel_size": kernel_size, "stride": stride}


# -- embedding -------------------------------------------------------------

def _gen_embedding(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    V: int = config["V"]
    S: int = config["S"]
    D: int = config["D"]
    B: int = config["B"]

    indices = torch.randint(0, V, (B, S), device=device)
    weight = torch.randn(V, D, dtype=dt, device=device)
    return {"indices": indices, "weight": weight}


# -- positional encoding (RoPE) -------------------------------------------

def _gen_rope(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    B: int = config["B"]
    H: int = config["H"]
    S: int = config["S"]
    D: int = config["D"]

    Q = torch.randn(B, H, S, D, dtype=dt, device=device)
    K = torch.randn(B, H, S, D, dtype=dt, device=device)
    return {"Q": Q, "K": K}


# -- fused matmul / fused gemm (Model-style) ------------------------------

def _gen_fused_matmul(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    B: int = config["B"]
    in_features: int = config["in_features"]
    return {"input": torch.randn(B, in_features, dtype=dt, device=device)}


def _gen_fused_gemm(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    B: int = config["B"]
    in_features: int = config["in_features"]
    return {"input": torch.randn(B, in_features, dtype=dt, device=device)}


# -- fused conv (Model-style) ---------------------------------------------

def _gen_fused_conv(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    B: int = config["B"]
    C_in: int = config["C_in"]
    H: int = config["H"]
    W: int = config["W"]

    if "D" in config:
        D: int = config["D"]
        return {"input": torch.randn(B, C_in, D, H, W, dtype=dt, device=device)}

    return {"input": torch.randn(B, C_in, H, W, dtype=dt, device=device)}


# -- model (CNN / Transformer / RNN / Other) -------------------------------

def _gen_model(config: Dict[str, Any], dtype_str: str, device: str) -> Dict[str, Any]:
    dt = _get_dtype(dtype_str)
    B: int = config.get("batch", config.get("B", 1))
    model_type: str = config.get("model_type", "")

    if model_type == "cnn" or "C_in" in config:
        C: int = config.get("C_in", 3)
        H: int = config.get("H", 224)
        W: int = config.get("W", 224)
        return {"input": torch.randn(B, C, H, W, dtype=dt, device=device)}

    if model_type == "transformer" or "hidden_size" in config:
        seq_len: int = config.get("seq_len", 128)
        hidden_size: int = config.get("hidden_size", 512)
        return {"input": torch.randn(B, seq_len, hidden_size, dtype=dt, device=device)}

    if model_type == "rnn" or "input_size" in config:
        seq_len = config.get("seq_len", 128)
        input_size: int = config.get("input_size", 128)
        return {"input": torch.randn(B, seq_len, input_size, dtype=dt, device=device)}

    # Fallback: read shape directly from config if provided
    shape: Optional[List[int]] = config.get("shape", None)
    if shape is not None:
        return {"input": torch.randn(*shape, dtype=dt, device=device)}

    # Last resort: single vector
    in_features: int = config.get("in_features", 512)
    return {"input": torch.randn(B, in_features, dtype=dt, device=device)}


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

GENERATORS: Dict[str, Optional[Callable[..., Dict[str, Any]]]] = {
    "matmul": _gen_matmul,
    "conv": _gen_conv,
    "attention": _gen_attention,
    "normalization": _gen_normalization,
    "activation": _gen_activation,
    "elementwise": _gen_elementwise,
    "reduction": _gen_reduction,
    "loss": _gen_loss,
    "cumulative": _gen_cumulative,
    "pooling": _gen_pooling,
    "embedding": _gen_embedding,
    "positional_encoding": _gen_rope,
    "dropout": _gen_activation,
    "fused_matmul": _gen_fused_matmul,
    "fused_gemm": _gen_fused_gemm,
    "fused_conv": _gen_fused_conv,
    "fused_conv2d": _gen_fused_conv,
    "fused_conv3d": _gen_fused_conv,
    "fused_convtranspose": _gen_fused_conv,
    "fused_convtranspose2d": _gen_fused_conv,
    "fused_convtranspose3d": _gen_fused_conv,
    "model_cnn": _gen_model,
    "model_transformer": _gen_model,
    "model_rnn": _gen_model,
    "model_other": _gen_model,
    "specialized": None,  # must use local override
    "quantization": _gen_matmul,  # same shape as matmul
}


# ---------------------------------------------------------------------------
# Override resolution
# ---------------------------------------------------------------------------

def resolve_input_gen(
    kernel_dir: Path,
    category: str,
) -> Callable[..., Dict[str, Any]]:
    """Return the appropriate ``generate`` function for a kernel.

    Resolution order:

    1. If ``kernel_dir/input_gen.py`` exists, dynamically import it and
       return its ``generate`` callable.
    2. Otherwise look up :data:`GENERATORS` by *category*.
    3. If the generator is ``None`` (e.g. ``"specialized"``), raise
       :class:`ValueError` because the kernel must provide its own
       ``input_gen.py``.

    Parameters
    ----------
    kernel_dir:
        Path to the kernel directory (may contain a local ``input_gen.py``).
    category:
        The kernel's category string as listed in :data:`GENERATORS`.

    Returns
    -------
    Callable
        A function with the same signature as :func:`generate`.
    """
    local_input_gen = Path(kernel_dir) / "input_gen.py"
    if local_input_gen.is_file():
        spec = importlib.util.spec_from_file_location(
            "input_gen_override", str(local_input_gen)
        )
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["input_gen_override"] = mod
            spec.loader.exec_module(mod)
            gen_fn = getattr(mod, "generate", None)
            if gen_fn is not None:
                return gen_fn

    gen = GENERATORS.get(category)
    if gen is None:
        raise ValueError(
            f"No input generator for category '{category}'. "
            f"Provide a local input_gen.py in {kernel_dir}."
        )
    # Return the central generate() which wraps the category dispatcher
    return generate


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate(
    kernel_name: str,
    category: str,
    config: Dict[str, Any],
    dtype: str,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate deterministic inputs for a kernel benchmark.

    Sets both CPU and CUDA random seeds for reproducibility, then
    dispatches to the appropriate category-specific generator.

    Parameters
    ----------
    kernel_name:
        Identifier for the kernel (used for diagnostics only).
    category:
        Category string matching a key in :data:`GENERATORS`.
    config:
        Shape / parameter dictionary consumed by the generator.
    dtype:
        Short dtype string (e.g. ``"fp16"``, ``"fp32"``).
    device:
        Torch device string.  Defaults to ``"cuda"``.
    seed:
        RNG seed for reproducibility.  Defaults to ``42``.

    Returns
    -------
    dict[str, torch.Tensor | int | ...]
        Named tensors (and possibly scalar parameters) ready for the
        kernel under test.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    gen_fn = GENERATORS.get(category)
    if gen_fn is None:
        raise ValueError(
            f"No built-in input generator for category '{category}' "
            f"(kernel '{kernel_name}'). Use resolve_input_gen() with a "
            f"local override instead."
        )

    return gen_fn(config, dtype, device)
