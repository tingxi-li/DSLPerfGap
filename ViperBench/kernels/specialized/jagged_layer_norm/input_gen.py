"""Local input generator for jagged_layer_norm."""
from __future__ import annotations
from typing import Any, Dict
import torch

def generate(kernel_name="", category="", config=None, dtype="fp32", device="cuda", seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if config is None:
        config = {}
    DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}
    dt = DTYPE_MAP.get(dtype, torch.float32)
    B = config.get("B", 16)
    M = config.get("M", config.get("D", 64))
    max_seqlen = config.get("max_seqlen", 128)
    lengths = torch.randint(1, max_seqlen + 1, (B,))
    offsets = torch.zeros(B + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lengths, dim=0).to(device)
    total = offsets[-1].item()
    values = torch.randn(total, M, dtype=dt, device=device)
    return {"values": values, "offsets": offsets, "M": M}
