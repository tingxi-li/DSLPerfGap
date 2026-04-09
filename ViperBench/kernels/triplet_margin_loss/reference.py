"""Reference: triplet_margin_loss"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    a = inputs["anchor"]
    p = inputs["positive"]
    n = inputs["negative"]
    return {"output": F.triplet_margin_loss(a, p, n)}
