"""Reference: layer_norm"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    w = inputs["weight"]
    b = inputs["bias"]
    return {"output": F.layer_norm(x, x.shape[-1:], w, b)}
