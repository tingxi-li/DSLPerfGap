"""Reference: hinge_loss"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pred = inputs["prediction"]
    target = inputs["target"]
    return {"output": torch.mean(torch.clamp(1 - pred * target, min=0))}
