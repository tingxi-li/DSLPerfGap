"""Reference: swiglu"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    x1, x2 = x.chunk(2, dim=-1)
    return {"output": x1 * F.silu(x2)}
