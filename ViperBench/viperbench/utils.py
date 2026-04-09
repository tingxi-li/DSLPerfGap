"""Shared utilities for ViperBench."""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from difflib import SequenceMatcher
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

import torch

# ---------------------------------------------------------------------------
# Package-relative paths
# ---------------------------------------------------------------------------
CONFIGS_DIR: Path = Path(__file__).resolve().parent.parent / "configs"
RESULTS_DIR: Path = Path(__file__).resolve().parent.parent / "results"

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------
DTYPE_MAP: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "int4": torch.int8,  # no native int4 dtype; use int8 storage, kernels handle packing
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}

BYTES_PER_ELEMENT: Dict[str, int] = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp64": 8,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int4": 1,  # packed 4-bit uses 1 byte per 2 elements, but stored as int8
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Look up a torch dtype by short string name.

    Parameters
    ----------
    dtype_str:
        One of the keys in :data:`DTYPE_MAP` (e.g. ``"fp16"``).

    Returns
    -------
    torch.dtype
        The corresponding torch dtype, defaulting to ``torch.float32`` if
        *dtype_str* is not recognised.
    """
    return DTYPE_MAP.get(dtype_str, torch.float32)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def _fuzzy_match_score(candidate: str, target: str) -> float:
    """Return a similarity ratio in [0, 1] between two GPU name strings."""
    return SequenceMatcher(None, candidate.lower(), target.lower()).ratio()


def load_hardware_config(path: Path) -> Dict[str, Any]:
    """Load a hardware specification JSON file.

    Parameters
    ----------
    path:
        Filesystem path to a ``*.json`` hardware config.

    Returns
    -------
    dict
        The parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def detect_hardware() -> Dict[str, Any]:
    """Auto-detect the current GPU and return its hardware spec.

    The function reads the device name reported by
    ``torch.cuda.get_device_name(0)`` and fuzzy-matches it against every
    JSON file under ``configs/hardware/``.  If a sufficiently close match
    is found the full spec dict from that file is returned.  Otherwise a
    generic dict is synthesised from ``torch.cuda.get_device_properties``.

    Returns
    -------
    dict
        Hardware specification dictionary.
    """
    if not torch.cuda.is_available():
        return {"name": "cpu", "error": "No CUDA device available"}

    detected_name: str = torch.cuda.get_device_name(0)
    hw_dir: Path = CONFIGS_DIR / "hardware"

    best_score: float = 0.0
    best_config: Optional[Dict[str, Any]] = None

    if hw_dir.is_dir():
        for json_path in hw_dir.glob("*.json"):
            try:
                config = load_hardware_config(json_path)
            except (json.JSONDecodeError, OSError):
                continue
            config_name = config.get("name", "")
            score = _fuzzy_match_score(config_name, detected_name)
            if score > best_score:
                best_score = score
                best_config = config

    # Threshold chosen empirically: typical GPU names share most tokens when
    # they really refer to the same card.
    if best_config is not None and best_score >= 0.6:
        return best_config

    # Fallback: build a generic spec from torch device properties.
    props = torch.cuda.get_device_properties(0)
    return {
        "name": detected_name,
        "compute_capability": f"sm_{props.major}{props.minor}",
        "compute_characteristics": {
            "sm_count": props.multi_processor_count,
        },
        "memory_hierarchy": {
            "memory_capacity_bytes": props.total_mem,
            "l2_cache_mb": round(props.L2_cache_size / (1024 * 1024), 1)
            if hasattr(props, "L2_cache_size")
            else None,
        },
    }


# ---------------------------------------------------------------------------
# Dynamic module import
# ---------------------------------------------------------------------------

def import_module_from_path(path: Path) -> ModuleType:
    """Dynamically import a Python module from a filesystem path.

    Parameters
    ----------
    path:
        Absolute or relative path to a ``.py`` file.

    Returns
    -------
    ModuleType
        The imported module object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ImportError
        If the module cannot be loaded.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Module path does not exist: {path}")
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(kernel_name: str, level: str = "INFO") -> logging.Logger:
    """Create a per-kernel logger writing to console and a log file.

    The log file is placed at ``results/<kernel_name>/eval.log``.  The
    ``results/<kernel_name>/`` directory is created automatically if it does
    not already exist.

    Parameters
    ----------
    kernel_name:
        Identifier used as both the logger name and the results sub-directory.
    level:
        Logging level string (e.g. ``"DEBUG"``, ``"INFO"``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(kernel_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers when called more than once for the same kernel.
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler
    log_dir = RESULTS_DIR / kernel_name
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "eval.log", encoding="utf-8")
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
