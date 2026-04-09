"""Reference: sdpa — scaled dot-product attention."""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    Q = inputs["Q"]
    K = inputs["K"]
    V = inputs["V"]
    is_causal = inputs.get("is_causal", False)
    output = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)
    return {"output": output}
