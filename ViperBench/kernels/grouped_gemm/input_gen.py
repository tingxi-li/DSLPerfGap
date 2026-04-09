"""Local input generator for grouped_gemm — produces A_list/B_list."""
from __future__ import annotations

import sys
import os
from typing import Any, Dict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from viperbench.utils import DTYPE_MAP


def generate(
    kernel_name: str = "",
    category: str = "",
    config: Dict[str, Any] = None,
    dtype: str = "fp32",
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if config is None:
        config = {}
    dt = DTYPE_MAP.get(dtype, torch.float32)
    M = config.get("M", 256)
    N = config.get("N", 256)
    K = config.get("K", 256)
    G = config.get("num_groups", 4)
    A_list = [torch.randn(M, K, dtype=dt, device=device) for _ in range(G)]
    B_list = [torch.randn(K, N, dtype=dt, device=device) for _ in range(G)]
    return {"A_list": A_list, "B_list": B_list}
