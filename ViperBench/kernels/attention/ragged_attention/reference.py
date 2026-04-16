"""Reference: ragged_attention"""
import torch
from torch.nn.attention.varlen import varlen_attn

def lengths_to_cu_seqlens(lengths: torch.Tensor) -> tuple[torch.Tensor, int]:
    # lengths: [B], int32/int64
    lengths = lengths.to(torch.int32)
    cu = torch.empty(lengths.numel() + 1, device=lengths.device, dtype=torch.int32)
    cu[0] = 0
    cu[1:] = torch.cumsum(lengths, dim=0)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    return cu, max_len

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Packed inputs, not padded dense tensors
    # Q: [total_q_tokens, H, D]
    # K: [total_k_tokens, H, D]
    # V: [total_k_tokens, H, Dv]
    Q = inputs["Q"]
    K = inputs["K"]
    V = inputs["V"]

    q_lens = inputs["q_lens"]   # [B]
    k_lens = inputs.get("k_lens", q_lens)

    cu_q, max_q = lengths_to_cu_seqlens(q_lens)
    cu_k, max_k = lengths_to_cu_seqlens(k_lens)

    causal = inputs.get("is_causal", False)
    window_size = (-1, 0) if causal else (-1, -1)

    out = varlen_attn(
        query=Q,
        key=K,
        value=V,
        cu_seq_q=cu_q,
        cu_seq_k=cu_k,
        max_q=max_q,
        max_k=max_k,
        window_size=window_size,
    )
    return {"output": out}