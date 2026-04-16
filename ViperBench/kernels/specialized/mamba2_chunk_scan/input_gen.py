"""Local input generator for mamba2_chunk_scan."""
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
    # Ensure seqlen == nchunks * chunk_size
    seqlen = nchunks * chunk_size
    return {
        "cb": torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size, dtype=dt_type, device=device),
        "x": torch.randn(batch, seqlen, nheads, dhead, dtype=dt_type, device=device),
        "dt": torch.randn(batch, nheads, nchunks, chunk_size, dtype=dt_type, device=device),
        "dA_cumsum": torch.rand(batch, nheads, nchunks, chunk_size, dtype=dt_type, device=device),
        "C": torch.randn(batch, seqlen, ngroups, dstate, dtype=dt_type, device=device),
        "prev_states": torch.randn(batch, nchunks, nheads, dhead, dstate, dtype=dt_type, device=device),
        "D": torch.randn(nheads, dtype=dt_type, device=device),
    }
