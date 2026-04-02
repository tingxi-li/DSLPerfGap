"""
Test case generator for KernelBench evaluation.

Generates TestCaseSpec instances for 297 kernels across 22 categories,
producing different input shapes/dtypes/layouts based on category.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import (
    DEFAULT_DTYPES,
    DEFAULT_LAYOUTS,
    DEFAULT_SHAPE_FAMILIES,
    DEFAULT_SIZE_BUCKETS,
    DEFAULT_VALUE_DISTS,
    FLOAT_DTYPES,
    FLOAT_ONLY_CATEGORIES,
    INT_DTYPES,
    SIZE_BUCKET_MAP,
    Layout,
    ShapeFamily,
    TestCaseSpec,
    ValueDist,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Category loading
# ═══════════════════════════════════════════════════════════════════════════════

_CATEGORIES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "categories", "categories.json"
)


def load_categories() -> Dict[str, List[str]]:
    """Load categories.json. Returns {category_name: [kernel_paths]}."""
    with open(_CATEGORIES_PATH, "r") as f:
        return json.load(f)


def get_kernel_category(kernel_path: str, categories: Dict[str, List[str]]) -> str:
    """Look up which category a kernel belongs to."""
    for category, paths in categories.items():
        if kernel_path in paths:
            return category
    return "specialized"  # fallback


# ═══════════════════════════════════════════════════════════════════════════════
# Module introspection
# ═══════════════════════════════════════════════════════════════════════════════


def get_default_input_info(module) -> Dict[str, Any]:
    """
    Call module.get_inputs() and inspect returned tensors.
    Returns info about shapes, dtypes, ndim, nbytes.
    Also calls get_init_inputs() if available.
    """
    info: Dict[str, Any] = {
        "inputs": [],
        "init_inputs": [],
        "total_bytes": 0,
    }

    try:
        inputs = module.get_inputs()
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                entry = {
                    "shape": list(inp.shape),
                    "dtype": inp.dtype,
                    "ndim": inp.ndim,
                    "nbytes": inp.nelement() * inp.element_size(),
                }
                info["inputs"].append(entry)
                info["total_bytes"] += entry["nbytes"]
            else:
                info["inputs"].append({"value": inp, "type": type(inp).__name__})
    except Exception:
        pass

    try:
        if hasattr(module, "get_init_inputs"):
            init_inputs = module.get_init_inputs()
            for inp in init_inputs:
                if isinstance(inp, torch.Tensor):
                    info["init_inputs"].append({
                        "shape": list(inp.shape),
                        "dtype": inp.dtype,
                    })
                else:
                    info["init_inputs"].append({"value": inp, "type": type(inp).__name__})
    except Exception:
        pass

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════


def _dtype_itemsize(dtype: torch.dtype) -> int:
    """Return bytes per element for a dtype."""
    return torch.tensor([], dtype=dtype).element_size()


def _num_elements(target_bytes: int, dtype: torch.dtype, num_tensors: int = 1) -> int:
    """Total elements per tensor to reach target_bytes across num_tensors tensors."""
    itemsize = _dtype_itemsize(dtype)
    total_elements = target_bytes // itemsize
    return max(total_elements // num_tensors, 1)


def _isqrt(n: int) -> int:
    """Integer square root."""
    return max(1, int(math.isqrt(max(n, 1))))


# ═══════════════════════════════════════════════════════════════════════════════
# Shape generators — elementwise
# ═══════════════════════════════════════════════════════════════════════════════


def generate_shapes_elementwise(
    target_bytes: int,
    shape_family: ShapeFamily,
    dtype: torch.dtype,
    num_inputs: int = 1,
) -> Optional[Tuple[List[Tuple[int, ...]], str]]:
    """
    For activation, elementwise, dropout categories.
    Scale a single tensor to target bytes.
    """
    N = _num_elements(target_bytes, dtype, num_inputs)

    if shape_family == ShapeFamily.FLAT_1D:
        shapes = [(N,)]
        desc = f"1D flat: ({N},)"

    elif shape_family == ShapeFamily.SQUARE_2D:
        side = _isqrt(N)
        shapes = [(side, side)]
        desc = f"2D square: ({side}, {side})"

    elif shape_family == ShapeFamily.TALL_SKINNY:
        cols = 64
        rows = max(N // cols, 1)
        shapes = [(rows, cols)]
        desc = f"2D tall-skinny: ({rows}, {cols})"

    elif shape_family == ShapeFamily.SHORT_WIDE:
        rows = 64
        cols = max(N // rows, 1)
        shapes = [(rows, cols)]
        desc = f"2D short-wide: ({rows}, {cols})"

    elif shape_family == ShapeFamily.NCHW_4D:
        channels = 64
        remaining = max(N // channels, 1)
        hw = _isqrt(remaining)
        # Find reasonable batch so batch * channels * hw * hw ~ N
        batch = max(remaining // (hw * hw), 1)
        shapes = [(batch, channels, hw, hw)]
        desc = f"4D NCHW: ({batch}, {channels}, {hw}, {hw})"

    elif shape_family == ShapeFamily.NON_POWER_OF_TWO:
        # Use prime-ish dimensions
        cols = 4093
        rows = max(N // cols, 1)
        if rows == 0:
            rows = 1
        shapes = [(rows, cols)]
        desc = f"Non-pow2: ({rows}, {cols})"

    elif shape_family == ShapeFamily.MISALIGNED:
        cols = 4093
        rows = max(N // cols, 1)
        if rows < 1:
            rows = 1
        # Ensure neither dim is divisible by 128
        if rows % 128 == 0:
            rows = rows - 1 if rows > 1 else 1
        shapes = [(rows, cols)]
        desc = f"Misaligned: ({rows}, {cols})"
    else:
        return None

    return [s for s in shapes for _ in range(num_inputs)], desc


# ═══════════════════════════════════════════════════════════════════════════════
# Shape generators — matmul
# ═══════════════════════════════════════════════════════════════════════════════


def generate_shapes_matmul(
    target_bytes: int,
    shape_family: ShapeFamily,
    dtype: torch.dtype,
) -> Optional[Tuple[List[Tuple[int, ...]], str]]:
    """
    For matmul, fused_gemm, fused_matmul, quantization.
    Need A(M,K) x B(K,N) -> C(M,N). Total bytes ~ (M*K + K*N + M*N) * itemsize.
    """
    itemsize = _dtype_itemsize(dtype)

    if shape_family == ShapeFamily.SQUARE_2D:
        # M=K=N, total = 3 * M^2 * itemsize
        M = _isqrt(target_bytes // (3 * itemsize))
        M = max(M, 1)
        shapes = [(M, M), (M, M)]
        desc = f"Square matmul: M=K=N={M}"

    elif shape_family == ShapeFamily.TALL_SKINNY:
        # M >> K, N small. K=256, N=64
        K, N = 256, 64
        # M*K + K*N + M*N ~ target
        # M*(K+N) + K*N ~ target
        M = max((target_bytes // itemsize - K * N) // (K + N), 1)
        shapes = [(M, K), (K, N)]
        desc = f"Tall-skinny matmul: M={M}, K={K}, N={N}"

    elif shape_family == ShapeFamily.SHORT_WIDE:
        # M small, N >> K. M=64, K=256
        M, K = 64, 256
        N = max((target_bytes // itemsize - M * K) // (K + M), 1)
        shapes = [(M, K), (K, N)]
        desc = f"Short-wide matmul: M={M}, K={K}, N={N}"

    elif shape_family == ShapeFamily.FLAT_1D:
        # Interpret as matrix-vector: A(M,K) x B(K,)
        K = 256
        M = max((target_bytes // itemsize - K) // (K + 1), 1)
        shapes = [(M, K), (K,)]
        desc = f"Mat-vec: M={M}, K={K}"

    elif shape_family == ShapeFamily.NCHW_4D:
        # Batched matmul: (B, M, K) x (B, K, N)
        M = _isqrt(target_bytes // (6 * itemsize))
        M = max(M, 1)
        B = max(target_bytes // (3 * M * M * itemsize), 1)
        shapes = [(B, M, M), (B, M, M)]
        desc = f"Batched matmul: B={B}, M=K=N={M}"

    elif shape_family == ShapeFamily.NON_POWER_OF_TWO:
        # M,K,N all prime-ish
        M = 1023
        K = 509
        N = max((target_bytes // itemsize - M * K) // (K + M), 1)
        shapes = [(M, K), (K, N)]
        desc = f"Non-pow2 matmul: M={M}, K={K}, N={N}"

    elif shape_family == ShapeFamily.MISALIGNED:
        M = 1023
        K = 511
        N = max((target_bytes // itemsize - M * K) // (K + M), 1)
        if N % 128 == 0:
            N = N - 1 if N > 1 else 1
        shapes = [(M, K), (K, N)]
        desc = f"Misaligned matmul: M={M}, K={K}, N={N}"

    else:
        return None

    return shapes, desc


# ═══════════════════════════════════════════════════════════════════════════════
# Shape generators — conv
# ═══════════════════════════════════════════════════════════════════════════════


def generate_shapes_conv(
    target_bytes: int,
    shape_family: ShapeFamily,
    dtype: torch.dtype,
    spatial_dims: int = 2,
) -> Optional[Tuple[List[Tuple[int, ...]], str]]:
    """
    For conv, fused_conv, fused_convtranspose, normalization, pooling.
    Fix channels C=64. Scale batch and spatial dims.
    Only NCHW_4D, SQUARE_2D, NON_POWER_OF_TWO, MISALIGNED apply.
    """
    itemsize = _dtype_itemsize(dtype)
    C = 64

    if shape_family not in (
        ShapeFamily.NCHW_4D,
        ShapeFamily.SQUARE_2D,
        ShapeFamily.NON_POWER_OF_TWO,
        ShapeFamily.MISALIGNED,
    ):
        return None  # SKIPPED for unsupported shape families

    # Total elements ~ target_bytes / itemsize
    total_elems = max(target_bytes // itemsize, 1)

    if shape_family == ShapeFamily.NCHW_4D:
        # (B, C, H, W, ...) with spatial_dims spatial dimensions
        # B * C * H^spatial_dims = total_elems
        spatial_total = max(total_elems // C, 1)
        hw = max(int(spatial_total ** (1.0 / (spatial_dims + 1))), 1)
        batch = max(spatial_total // (hw ** spatial_dims), 1)
        shape = (batch, C) + (hw,) * spatial_dims
        shapes = [shape]
        desc = f"NCHW conv: {shape}"

    elif shape_family == ShapeFamily.SQUARE_2D:
        # Interpret as H=W (for 2D conv)
        spatial_total = max(total_elems // C, 1)
        hw = _isqrt(spatial_total)
        batch = max(spatial_total // (hw * hw), 1)
        shape = (batch, C) + (hw,) * spatial_dims
        shapes = [shape]
        desc = f"Square-spatial conv: {shape}"

    elif shape_family == ShapeFamily.NON_POWER_OF_TWO:
        spatial_total = max(total_elems // C, 1)
        hw = max(int(spatial_total ** (1.0 / (spatial_dims + 1))), 1)
        # Make hw prime-ish
        if hw > 10:
            hw = hw | 1  # make odd
            if hw % 3 == 0:
                hw += 2
        batch = max(spatial_total // (hw ** spatial_dims), 1)
        shape = (batch, C) + (hw,) * spatial_dims
        shapes = [shape]
        desc = f"Non-pow2 conv: {shape}"

    elif shape_family == ShapeFamily.MISALIGNED:
        spatial_total = max(total_elems // C, 1)
        hw = max(int(spatial_total ** (1.0 / (spatial_dims + 1))), 1)
        # Ensure not divisible by 128
        if hw % 128 == 0:
            hw = hw - 1 if hw > 1 else 1
        batch = max(spatial_total // (hw ** spatial_dims), 1)
        shape = (batch, C) + (hw,) * spatial_dims
        shapes = [shape]
        desc = f"Misaligned conv: {shape}"
    else:
        return None

    return shapes, desc


# ═══════════════════════════════════════════════════════════════════════════════
# Shape generators — attention
# ═══════════════════════════════════════════════════════════════════════════════


def generate_shapes_attention(
    target_bytes: int,
    shape_family: ShapeFamily,
    dtype: torch.dtype,
) -> Optional[Tuple[List[Tuple[int, ...]], str]]:
    """
    For attention. Q,K,V are (B, H, S, D). Fix H=8, D=64. Scale B and S.
    """
    itemsize = _dtype_itemsize(dtype)
    H, D = 8, 64

    # 3 tensors (Q, K, V) each of shape (B, H, S, D)
    # total = 3 * B * H * S * D * itemsize
    total_elems = max(target_bytes // itemsize, 1)
    per_tensor = max(total_elems // 3, 1)
    # per_tensor = B * H * S * D => B * S = per_tensor / (H * D)
    BS = max(per_tensor // (H * D), 1)

    if shape_family == ShapeFamily.FLAT_1D:
        # 1D not natural for attention but use S=BS, B=1
        B, S = 1, BS
    elif shape_family == ShapeFamily.SQUARE_2D:
        S = _isqrt(BS)
        B = max(BS // S, 1)
    elif shape_family == ShapeFamily.TALL_SKINNY:
        B = 1
        S = BS
    elif shape_family == ShapeFamily.SHORT_WIDE:
        S = 64
        B = max(BS // S, 1)
    elif shape_family == ShapeFamily.NCHW_4D:
        S = _isqrt(BS)
        B = max(BS // S, 1)
    elif shape_family == ShapeFamily.NON_POWER_OF_TWO:
        S = _isqrt(BS)
        if S > 10:
            S = S | 1
        B = max(BS // S, 1)
    elif shape_family == ShapeFamily.MISALIGNED:
        S = _isqrt(BS)
        if S % 128 == 0:
            S = S - 1 if S > 1 else 1
        B = max(BS // S, 1)
    else:
        return None

    qkv_shape = (B, H, S, D)
    shapes = [qkv_shape, qkv_shape, qkv_shape]
    desc = f"Attention Q/K/V: {qkv_shape}"
    return shapes, desc


# ═══════════════════════════════════════════════════════════════════════════════
# Shape generators — loss
# ═══════════════════════════════════════════════════════════════════════════════


def generate_shapes_loss(
    target_bytes: int,
    shape_family: ShapeFamily,
    dtype: torch.dtype,
) -> Optional[Tuple[List[Tuple[int, ...]], str]]:
    """
    For loss. Two matching tensors (predictions, targets).
    """
    result = generate_shapes_elementwise(target_bytes, shape_family, dtype, num_inputs=2)
    if result is None:
        return None
    shapes, desc = result
    return shapes, f"Loss {desc}"


# ═══════════════════════════════════════════════════════════════════════════════
# Shape generators — reduction / cumulative
# ═══════════════════════════════════════════════════════════════════════════════


def generate_shapes_reduction(
    target_bytes: int,
    shape_family: ShapeFamily,
    dtype: torch.dtype,
) -> Optional[Tuple[List[Tuple[int, ...]], str]]:
    """
    For reduction, cumulative. Single input tensor, same as elementwise.
    """
    result = generate_shapes_elementwise(target_bytes, shape_family, dtype, num_inputs=1)
    if result is None:
        return None
    shapes, desc = result
    return shapes, f"Reduction {desc}"


# ═══════════════════════════════════════════════════════════════════════════════
# Shape generators — model
# ═══════════════════════════════════════════════════════════════════════════════


def generate_shapes_model(
    target_bytes: int,
    shape_family: ShapeFamily,
    dtype: torch.dtype,
) -> Optional[Tuple[List[Tuple[int, ...]], str]]:
    """
    For model_* categories. Only scale batch size.
    Input is (B, 3, 224, 224) for CNNs.
    Only NCHW_4D supported; others -> None (SKIPPED).
    """
    if shape_family != ShapeFamily.NCHW_4D:
        return None

    itemsize = _dtype_itemsize(dtype)
    # single_image = 3 * 224 * 224 elements
    single_image_elems = 3 * 224 * 224
    single_image_bytes = single_image_elems * itemsize
    batch = max(target_bytes // single_image_bytes, 1)
    shape = (batch, 3, 224, 224)
    shapes = [shape]
    desc = f"Model input: {shape}"
    return shapes, desc


# ═══════════════════════════════════════════════════════════════════════════════
# Category -> generator dispatch
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORY_TO_GENERATOR = {
    "activation": generate_shapes_elementwise,
    "elementwise": generate_shapes_elementwise,
    "dropout": generate_shapes_elementwise,
    "matmul": generate_shapes_matmul,
    "fused_gemm": generate_shapes_matmul,
    "fused_matmul": generate_shapes_matmul,
    "quantization": generate_shapes_matmul,
    "conv": generate_shapes_conv,
    "fused_conv": generate_shapes_conv,
    "fused_convtranspose": generate_shapes_conv,
    "normalization": generate_shapes_conv,
    "pooling": generate_shapes_conv,
    "attention": generate_shapes_attention,
    "loss": generate_shapes_loss,
    "reduction": generate_shapes_reduction,
    "cumulative": generate_shapes_reduction,
    "embedding": generate_shapes_elementwise,
    "specialized": generate_shapes_elementwise,
    "model_cnn": generate_shapes_model,
    "model_transformer": generate_shapes_model,
    "model_rnn": generate_shapes_model,
    "model_other": generate_shapes_model,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor creation
# ═══════════════════════════════════════════════════════════════════════════════


def make_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    value_dist: ValueDist,
    layout: Layout,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create tensor with specified properties.
    Handles value distributions and memory layouts.
    """
    is_int = dtype in INT_DTYPES

    # --- Value distribution ---
    if value_dist == ValueDist.UNIFORM:
        if is_int:
            t = torch.randint(0, 100, shape, dtype=dtype, device=device)
        else:
            t = torch.rand(shape, dtype=dtype, device=device)

    elif value_dist == ValueDist.NORMAL:
        if is_int:
            t = torch.randint(-50, 50, shape, dtype=dtype, device=device)
        else:
            t = torch.randn(shape, dtype=dtype, device=device)

    elif value_dist == ValueDist.ZEROS:
        t = torch.zeros(shape, dtype=dtype, device=device)

    elif value_dist == ValueDist.LARGE:
        if is_int:
            t = torch.randint(0, 1_000_000, shape, dtype=dtype, device=device)
        else:
            t = torch.rand(shape, dtype=dtype, device=device) * 1e6

    elif value_dist == ValueDist.SMALL:
        if is_int:
            t = torch.randint(0, 2, shape, dtype=dtype, device=device)
        else:
            t = torch.rand(shape, dtype=dtype, device=device) * 1e-6

    elif value_dist == ValueDist.NAN_INF:
        if is_int:
            # No NaN/Inf for int, just random
            t = torch.randint(0, 100, shape, dtype=dtype, device=device)
        else:
            t = torch.rand(shape, dtype=dtype, device=device)
            # Inject ~1% NaN and ~1% Inf
            mask_nan = torch.rand(shape, device=device) < 0.005
            mask_inf = torch.rand(shape, device=device) < 0.005
            mask_neginf = torch.rand(shape, device=device) < 0.005
            t[mask_nan] = float("nan")
            t[mask_inf] = float("inf")
            t[mask_neginf] = float("-inf")
    else:
        t = torch.rand(shape, dtype=dtype, device=device)

    # --- Layout ---
    if layout == Layout.CONTIGUOUS:
        pass  # already contiguous

    elif layout == Layout.STRIDED:
        # Pad each dim by 1 and slice back to create non-contiguous strides
        if len(shape) >= 1:
            padded_shape = tuple(s + 1 for s in shape)
            if is_int:
                padded = torch.randint(0, 100, padded_shape, dtype=dtype, device=device)
            else:
                padded = torch.rand(padded_shape, dtype=dtype, device=device)
            slices = tuple(slice(0, s) for s in shape)
            strided = padded[slices]
            # Copy values from t
            strided.copy_(t)
            t = strided

    elif layout == Layout.TRANSPOSED:
        if len(shape) >= 2:
            # Transpose last two dims then make a view
            t = t.transpose(-2, -1).contiguous().transpose(-2, -1)
        # For 1D, transposed is same as contiguous

    elif layout == Layout.BROADCASTED:
        if len(shape) >= 2:
            # Create a smaller tensor and expand
            reduced_shape = (1,) + shape[1:]
            if is_int:
                small = torch.randint(0, 100, reduced_shape, dtype=dtype, device=device)
            else:
                small = torch.rand(reduced_shape, dtype=dtype, device=device)
            t = small.expand(shape)
        # For 1D, broadcasted is same as contiguous

    return t


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def generate_test_cases(
    kernel_path: str,
    category: str,
    module: Any = None,
    size_buckets: Optional[List[str]] = None,
    shape_families: Optional[List[ShapeFamily]] = None,
    dtypes: Optional[List[torch.dtype]] = None,
    layouts: Optional[List[Layout]] = None,
    value_dists: Optional[List[ValueDist]] = None,
) -> List[TestCaseSpec]:
    """
    Main entry point. Generate cross-product of test cases,
    filtering impossible combos.

    Returns list of TestCaseSpec for the given kernel.
    """
    if size_buckets is None:
        size_buckets = DEFAULT_SIZE_BUCKETS
    if shape_families is None:
        shape_families = list(DEFAULT_SHAPE_FAMILIES)
    if dtypes is None:
        dtypes = list(DEFAULT_DTYPES)
    if layouts is None:
        layouts = list(DEFAULT_LAYOUTS)
    if value_dists is None:
        value_dists = list(DEFAULT_VALUE_DISTS)

    generator = CATEGORY_TO_GENERATOR.get(category, generate_shapes_elementwise)

    # Filter impossible dtype combos
    is_float_only = category in FLOAT_ONLY_CATEGORIES
    is_embedding = category == "embedding"

    cases: List[TestCaseSpec] = []

    for bucket_label in size_buckets:
        target_bytes = SIZE_BUCKET_MAP.get(bucket_label)
        if target_bytes is None:
            continue

        for sf in shape_families:
            for dtype in dtypes:
                # Skip int dtypes for float-only categories
                if is_float_only and dtype in INT_DTYPES:
                    continue

                # For embedding, index tensors are int — skip float dtypes
                # for the index input (but the embedding table uses float).
                # We still generate specs; the runner handles dtype mapping.
                if is_embedding and dtype in INT_DTYPES:
                    # Embedding indices can be int — allow
                    pass

                # Check if the shape family is supported by the generator
                result = None
                try:
                    result = generator(target_bytes, sf, dtype)
                except Exception:
                    pass

                if result is None:
                    # Shape family not supported for this category — skip
                    continue

                for layout in layouts:
                    for vd in value_dists:
                        cases.append(
                            TestCaseSpec(
                                kernel_path=kernel_path,
                                category=category,
                                size_bucket=bucket_label,
                                shape_family=sf,
                                layout=layout,
                                value_dist=vd,
                                dtype=dtype,
                                target_bytes=target_bytes,
                            )
                        )

    return cases


# ═══════════════════════════════════════════════════════════════════════════════
# Input preparation from spec
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_inputs_from_spec(
    spec: TestCaseSpec,
    module: Any,
    device: str = "cuda",
) -> Optional[List]:
    """Create actual input tensors from a TestCaseSpec.

    Uses category-aware shape generators for the target size, then creates
    tensors with the specified dtype, value distribution, and layout.
    For stateful kernels (conv, norm, etc.), falls back to scaling the
    default inputs from the module.

    Returns a list of inputs (tensors + non-tensor args) or None if the
    shape family is not supported.
    """
    generator = CATEGORY_TO_GENERATOR.get(spec.category, generate_shapes_elementwise)
    result = None
    try:
        result = generator(spec.target_bytes, spec.shape_family, spec.dtype)
    except Exception:
        pass

    if result is None:
        return None

    shapes, _desc = result

    # For most categories: create tensors from generated shapes
    inputs = []
    for shape in shapes:
        t = make_tensor(shape, spec.dtype, spec.value_dist, spec.layout, device=device)
        inputs.append(t)

    return inputs
