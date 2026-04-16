"""Reference: nvfp4_gemm"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    A = inputs["A"]
    B = inputs["B"]
    return {"output": torch.matmul(A.float(), B.float())}
