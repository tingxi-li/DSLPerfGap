"""Reference: low_mem_dropout"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    return {"output": F.dropout(x, p=0.1, training=True)}
