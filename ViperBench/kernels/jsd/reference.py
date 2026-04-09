"""Reference: jsd"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    p = inputs["p"]
    q = inputs["q"]
    m = 0.5 * (p + q)
    return {"output": 0.5 * (F.kl_div(m.log(), p, reduction="batchmean") + F.kl_div(m.log(), q, reduction="batchmean"))}
