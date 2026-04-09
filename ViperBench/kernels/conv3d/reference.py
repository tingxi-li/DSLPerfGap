"""Reference: conv3d"""
import torch

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["input"]
    weight = inputs["weight"]
    bias = inputs.get("bias")
    stride = inputs.get("stride", 1)
    padding = inputs.get("padding", 0)
    dilation = inputs.get("dilation", 1)
    groups = inputs.get("groups", 1)
    output = torch.conv3d(x, weight, bias, stride, padding, dilation, groups)
    return {"output": output}
