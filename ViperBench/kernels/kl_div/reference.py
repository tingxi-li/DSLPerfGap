"""Reference: kl_div"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pred = inputs["prediction"]
    target = inputs["target"]
    return {"output": F.kl_div(F.log_softmax(pred, dim=-1), target, reduction="batchmean")}
