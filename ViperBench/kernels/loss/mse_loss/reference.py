"""Reference: mse_loss"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pred = inputs["prediction"]
    target = inputs["target"]
    return {"output": F.mse_loss(pred, target)}
