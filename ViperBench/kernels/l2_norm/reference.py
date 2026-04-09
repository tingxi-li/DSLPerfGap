"""Reference: l2_norm"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    return {"output": F.normalize(x, p=2, dim=-1)}
