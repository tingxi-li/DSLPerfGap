"""Reference: decoding_attention"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    Q = inputs["Q"]
    K = inputs["K"]
    V = inputs["V"]
    return {"output": F.scaled_dot_product_attention(Q, K, V)}
