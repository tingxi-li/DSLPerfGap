"""Reference: embedding"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    indices = inputs["indices"]
    weight = inputs["weight"]
    return {"output": F.embedding(indices, weight)}
