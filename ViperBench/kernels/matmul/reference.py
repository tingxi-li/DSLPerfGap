"""
Reference implementation for: matmul
Computes: C = A @ B (optionally batched, transposed)
"""
import torch


def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    A = inputs["A"]
    B = inputs["B"]
    C = torch.matmul(A, B)
    return {"C": C}
