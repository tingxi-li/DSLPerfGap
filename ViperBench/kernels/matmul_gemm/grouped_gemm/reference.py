"""Reference: grouped_gemm — multiple independent GEMMs."""
import torch

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    A_list = inputs["A_list"]  # list of tensors or stacked (G, M, K)
    B_list = inputs["B_list"]  # list of tensors or stacked (G, K, N)
    results = [torch.matmul(a, b) for a, b in zip(A_list, B_list)]
    return {"C_list": results}
