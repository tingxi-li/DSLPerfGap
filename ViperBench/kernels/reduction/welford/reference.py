"""Reference: welford"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    mean = x.mean(dim=-1)
    var = x.var(dim=-1)
    return {"mean": mean, "var": var}
