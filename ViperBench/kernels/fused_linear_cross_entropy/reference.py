"""Reference: fused_linear_cross_entropy"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    weight = inputs["weight"]
    target = inputs["target"]
    logits = F.linear(x, weight)
    return {"output": F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))}
