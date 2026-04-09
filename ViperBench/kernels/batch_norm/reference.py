"""Reference: batch_norm"""
import torch

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    device = x.device
    bn = torch.nn.BatchNorm2d(x.shape[1]).to(device).eval()
    bn.weight.data = inputs["weight"].to(device)
    bn.bias.data = inputs["bias"].to(device)
    return {"output": bn(x)}
