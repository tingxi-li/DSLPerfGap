"""Reference: mingpt_gelu"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    # historically: return {"output": 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3.0))))}
    return {"output": F.gelu(x, approximate="tanh")}
