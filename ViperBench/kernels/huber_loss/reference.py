"""Reference: huber_loss"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pred = inputs["prediction"]
    target = inputs["target"]
    return {"output": F.smooth_l1_loss(pred, target)}
