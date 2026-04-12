"""Reference: jagged_softmax — softmax over variable-length rows."""
import torch

def reference(inputs):
    values = inputs["values"]
    offsets = inputs["offsets"]
    M = inputs["M"]
    max_seqlen = inputs["max_seqlen"]
    B = offsets.shape[0] - 1
    padded = torch.full((B, max_seqlen, M), float("-inf"), device=values.device, dtype=values.dtype)
    for i in range(B):
        start = offsets[i].item()
        end = offsets[i + 1].item()
        padded[i, :end - start] = values[start:end]
    padded_softmax = torch.softmax(padded, dim=1)
    output = torch.zeros_like(values)
    for i in range(B):
        start = offsets[i].item()
        end = offsets[i + 1].item()
        output[start:end] = padded_softmax[i, :end - start]
    return {"output": output}
