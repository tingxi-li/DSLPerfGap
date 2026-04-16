"""Local input generator for ragged_attention (packed/variable-length sequences)."""
from __future__ import annotations
from typing import Any, Dict
import torch

DTYPE_MAP = {
    "fp16": torch.float16, "bf16": torch.bfloat16,
    "fp32": torch.float32, "fp64": torch.float64,
}


def generate(kernel_name="", category="", config=None, dtype="fp32", device="cuda", seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if config is None:
        config = {}

    dt = DTYPE_MAP.get(dtype, torch.float32)
    B = config.get("B", 4)
    H = config.get("H", 32)
    S = config.get("S", 1024)
    D = config.get("D", 128)

    # Generate variable-length sequences: each between S//2 and S
    lo = max(1, S // 2)
    q_lens = torch.randint(lo, S + 1, (B,), dtype=torch.int32, device=device)
    k_lens = q_lens.clone()  # same lengths for self-attention

    total_q = int(q_lens.sum().item())
    total_k = int(k_lens.sum().item())

    Q = torch.randn(total_q, H, D, dtype=dt, device=device)
    K = torch.randn(total_k, H, D, dtype=dt, device=device)
    V = torch.randn(total_k, H, D, dtype=dt, device=device)

    return {"Q": Q, "K": K, "V": V, "q_lens": q_lens, "k_lens": k_lens}
