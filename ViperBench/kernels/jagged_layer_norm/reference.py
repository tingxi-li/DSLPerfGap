"""Reference: jagged_layer_norm — layer norm on variable-length rows."""
import torch

EPSILON = 1e-6

def reference(inputs):
    values = inputs["values"]
    offsets = inputs["offsets"]
    M = inputs["M"]
    B = offsets.shape[0] - 1
    max_seqlen = 0
    for i in range(B):
        length = (offsets[i + 1] - offsets[i]).item()
        if length > max_seqlen:
            max_seqlen = length
    padded = torch.zeros(B, max_seqlen, M, device=values.device, dtype=values.dtype)
    mask = torch.zeros(B, max_seqlen, 1, device=values.device, dtype=values.dtype)
    for i in range(B):
        start = offsets[i].item()
        end = offsets[i + 1].item()
        length = end - start
        padded[i, :length] = values[start:end]
        mask[i, :length] = 1.0
    ragged_lengths = (offsets[1:] - offsets[:-1]).float().unsqueeze(1).unsqueeze(2) * M
    mean = (padded * mask).sum(dim=(1, 2), keepdim=True) / ragged_lengths
    normalized = (padded - mean) * mask
    variance = (normalized ** 2).sum(dim=(1, 2), keepdim=True) / ragged_lengths
    padded_ln = normalized / torch.sqrt(variance + EPSILON)
    output = torch.zeros_like(values)
    for i in range(B):
        start = offsets[i].item()
        end = offsets[i + 1].item()
        output[start:end] = padded_ln[i, :end - start]
    return {"output": output}
