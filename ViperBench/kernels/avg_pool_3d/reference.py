"""Reference: avg_pool_3d"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    ks = inputs.get("kernel_size", 3)
    st = inputs.get("stride", 2)
    return {"output": F.avg_pool3d(x, ks, st)}
