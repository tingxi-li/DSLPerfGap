"""Reference: group_norm"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    num_groups = inputs.get("num_groups", 8)
    w = inputs["weight"]
    b = inputs["bias"]
    return {"output": F.group_norm(x, num_groups, w, b)}
