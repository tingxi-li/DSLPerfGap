"""Local input generator for mamba2_chunk_state."""
from __future__ import annotations
from typing import Any, Dict
import torch

def generate(kernel_name="", category="", config=None, dtype="fp32", device="cuda", seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if config is None:
        config = {}
    DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}
    dt_type = DTYPE_MAP.get(dtype, torch.float32)
    batch = config.get("B", 1)
    seqlen = config.get("S", 1024)
    nheads = config.get("nheads", 64)
    ngroups = config.get("ngroups", 1)
    chunk_size = config.get("chunk_size", 256)
    dhead = config.get("dhead", config.get("D", 64))
    dstate = config.get("dstate", 128)
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    return {
        "B_mat": torch.rand(batch, seqlen, ngroups, dstate, dtype=dt_type, device=device),
        "x": torch.rand(batch, seqlen, nheads, dhead, dtype=dt_type, device=device),
        "dt": torch.rand(batch, nheads, nchunks, chunk_size, dtype=dt_type, device=device),
        "dA_cumsum": torch.rand(batch, nheads, nchunks, chunk_size, dtype=dt_type, device=device),
    }
