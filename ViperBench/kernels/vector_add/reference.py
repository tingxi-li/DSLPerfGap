"""Reference: vector_add"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    a = inputs["A"]
    b = inputs["B"]
    return {"output": a + b}
