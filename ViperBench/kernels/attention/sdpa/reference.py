"""Reference: sdpa — scaled dot-product attention."""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    Q = inputs["Q"]
    K = inputs["K"]
    V = inputs["V"]
    is_causal = inputs.get("is_causal", False)

    # GQA/MQA: expand K/V heads to match Q heads
    q_heads = Q.shape[1]
    kv_heads = K.shape[1]
    if kv_heads != q_heads and kv_heads > 1:
        n_rep = q_heads // kv_heads
        K = K.repeat_interleave(n_rep, dim=1)
        V = V.repeat_interleave(n_rep, dim=1)

    output = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)
    return {"output": output}
