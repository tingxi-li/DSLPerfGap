"""Reference: scalar_mul — element-wise multiplication by scalar."""
import torch

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    A = inputs["A"]
    s = inputs["s"]
    return {"output": A * s}
