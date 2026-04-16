"""Reference: flex_attention"""
import torch
from torch.nn.attention.flex_attention import flex_attention

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    Q = inputs["Q"]
    K = inputs["K"]
    V = inputs["V"]

    out = flex_attention(Q, K, V)
    return {"output": out}