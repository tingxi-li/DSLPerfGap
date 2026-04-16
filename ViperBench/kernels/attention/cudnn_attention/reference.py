"""Reference: cudnn_attention — cuDNN attention via sdpa backend selection."""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    Q = inputs["Q"]
    K = inputs["K"]
    V = inputs["V"]
    is_causal = bool(inputs.get("is_causal", False))

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=is_causal, enable_gqa=True
        )
    return {"output": output}
