"""Reference: gather_gemv — gather + matrix-vector product."""
import torch

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    weight = inputs["weight"]
    x = inputs["x"]
    indices = inputs["indices"]
    gathered = weight[indices]
    output = torch.matmul(gathered, x.unsqueeze(-1)).squeeze(-1)
    return {"output": output}
