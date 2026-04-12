"""Reference: jagged_sum — sum over variable-length rows."""
import torch

def reference(inputs):
    values = inputs["values"]
    offsets = inputs["offsets"]
    M = inputs["M"]
    max_seqlen = inputs["max_seqlen"]
    B = offsets.shape[0] - 1
    padded = torch.zeros(B, max_seqlen, M, device=values.device, dtype=values.dtype)
    for i in range(B):
        start = offsets[i].item()
        end = offsets[i + 1].item()
        padded[i, :end - start] = values[start:end]
    return {"output": padded.sum(dim=1)}
