"""
Constants, enums, and dataclasses for the evaluation system.
Based on EVALUATION_SPEC.md.
"""

import dataclasses
from enum import Enum
from typing import Any, List, Optional, Set

import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Status codes (Section 9)
# ═══════════════════════════════════════════════════════════════════════════════

class Status(str, Enum):
    PASS = "PASS"
    FAIL_CORRECTNESS = "FAIL_CORRECTNESS"
    FAIL_RUNTIME_ERROR = "FAIL_RUNTIME_ERROR"
    OOM = "OOM"
    NOT_SUPPORTED_BY_HARDWARE = "NOT_SUPPORTED_BY_HARDWARE"
    NOT_SUPPORTED_BY_KERNEL = "NOT_SUPPORTED_BY_KERNEL"
    NOT_SUPPORTED_BY_REFERENCE = "NOT_SUPPORTED_BY_REFERENCE"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark parameters (Section 3.2)
# ═══════════════════════════════════════════════════════════════════════════════

WARMUP_RUNS: int = 10
TIMED_RUNS: int = 50
TIMEOUT_SECONDS: int = 120


# ═══════════════════════════════════════════════════════════════════════════════
# Tolerance table (Section 4.3)
# Maps torch.dtype -> (tau_elem, tau_ratio)
# ═══════════════════════════════════════════════════════════════════════════════

TOLERANCES = {
    torch.int8:     (0.0, 0.0),
    torch.int16:    (0.0, 0.0),
    torch.int32:    (0.0, 0.0),
    torch.int64:    (0.0, 0.0),
    torch.float64:  (1e-12, 1e-9),
    torch.float32:  (1e-5, 1e-6),
    torch.bfloat16: (1e-2, 1e-3),
    torch.float16:  (1e-2, 1e-3),
}

EPS: float = 1e-12  # denominator stabilizer for relative error


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism (Section 4.2)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SEED: int = 42
DETERMINISM_RUNS: int = 5


# ═══════════════════════════════════════════════════════════════════════════════
# Size buckets (Section 5.1) — (label, target_bytes)
# ═══════════════════════════════════════════════════════════════════════════════

SIZE_BUCKETS = [
    ("<1GB",  512 * 1024**2),
    ("2GB",   2 * 1024**3),
    ("4GB",   4 * 1024**3),
    ("8GB",   8 * 1024**3),
    ("16GB",  16 * 1024**3),
    ("32GB",  32 * 1024**3),
    ("64GB",  64 * 1024**3),
]

SIZE_BUCKET_MAP = {label: nbytes for label, nbytes in SIZE_BUCKETS}

# Minimal acceptance defaults (Section 12): >= 3 size buckets
DEFAULT_SIZE_BUCKETS = ["<1GB", "2GB", "4GB"]


# ═══════════════════════════════════════════════════════════════════════════════
# Shape families (Section 5.2)
# ═══════════════════════════════════════════════════════════════════════════════

class ShapeFamily(str, Enum):
    FLAT_1D = "1d_flat"
    SQUARE_2D = "2d_square"
    TALL_SKINNY = "2d_tall_skinny"
    SHORT_WIDE = "2d_short_wide"
    NCHW_4D = "4d_nchw"
    NON_POWER_OF_TWO = "non_pow2"
    MISALIGNED = "misaligned"

# Minimal acceptance defaults (Section 12): >= 3 shape families
DEFAULT_SHAPE_FAMILIES = [
    ShapeFamily.FLAT_1D,
    ShapeFamily.SQUARE_2D,
    ShapeFamily.NCHW_4D,
]


# ═══════════════════════════════════════════════════════════════════════════════
# Layout types (Section 5.3)
# ═══════════════════════════════════════════════════════════════════════════════

class Layout(str, Enum):
    CONTIGUOUS = "contiguous"
    STRIDED = "strided"
    TRANSPOSED = "transposed"
    BROADCASTED = "broadcasted"

DEFAULT_LAYOUTS = [Layout.CONTIGUOUS]


# ═══════════════════════════════════════════════════════════════════════════════
# Value distributions (Section 5.4)
# ═══════════════════════════════════════════════════════════════════════════════

class ValueDist(str, Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    ZEROS = "zeros"
    LARGE = "large"
    SMALL = "small"
    NAN_INF = "nan_inf"

DEFAULT_VALUE_DISTS = [ValueDist.UNIFORM]


# ═══════════════════════════════════════════════════════════════════════════════
# Dtype coverage (Section 6)
# ═══════════════════════════════════════════════════════════════════════════════

EVAL_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.int32,
    torch.int64,
]

# Minimal acceptance defaults (Section 12): >= 2 dtypes
DEFAULT_DTYPES = [torch.float32, torch.float16]

# Float dtypes (for kernels that require float)
FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
INT_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64}

# Categories that only work with float dtypes
FLOAT_ONLY_CATEGORIES = {
    "conv", "fused_conv", "fused_convtranspose",
    "normalization", "pooling", "attention",
    "activation", "dropout",
    "model_cnn", "model_transformer", "model_rnn", "model_other",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Test case specification
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class TestCaseSpec:
    kernel_path: str
    category: str
    size_bucket: str
    shape_family: ShapeFamily
    layout: Layout
    value_dist: ValueDist
    dtype: torch.dtype
    target_bytes: int


# ═══════════════════════════════════════════════════════════════════════════════
# Test case result (Section 10 JSON schema)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class TestCaseResult:
    op_name: str
    backend: str
    reference_backend: str

    device: str
    dtype_in: str
    dtype_out: str
    dtype_accum: str

    shape: list
    layout: str

    input_bytes_total: int
    output_bytes_total: int
    theoretical_io_bytes: int

    warmup_runs: int
    timed_runs: int

    latency_ms_mean: Optional[float] = None
    latency_ms_median: Optional[float] = None
    latency_ms_std: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    latency_ms_min: Optional[float] = None

    bandwidth_gbps: Optional[float] = None

    max_abs_error: Optional[float] = None
    mean_abs_error: Optional[float] = None
    max_rel_error: Optional[float] = None
    bad_elem_ratio: Optional[float] = None
    correctness_pass: Optional[bool] = None

    status: Status = Status.SKIPPED
    notes: str = ""
