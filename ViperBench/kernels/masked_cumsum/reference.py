"""Reference: masked_cumsum"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    mask = inputs["mask"]
    return {"output": torch.cumsum(x * mask, dim=1)}
