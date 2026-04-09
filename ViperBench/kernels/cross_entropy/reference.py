"""Reference: cross_entropy"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pred = inputs["prediction"]
    target = inputs["target"]
    return {"output": F.cross_entropy(pred, target)}
