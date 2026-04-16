"""Reference: mean_reduction"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    dim = inputs.get("reduce_dim", -1)
    return {"output": torch.mean(x, dim=dim)}
