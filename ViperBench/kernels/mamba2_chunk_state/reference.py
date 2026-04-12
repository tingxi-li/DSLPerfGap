"""Reference: mamba2_chunk_state — Mamba2 chunk state forward pass."""
import torch
import torch.nn.functional as F

def reference(inputs):
    B_mat = inputs["B_mat"]
    x = inputs["x"]
    dt = inputs["dt"]
    dA_cumsum = inputs["dA_cumsum"]

    batch, seqlen, nheads, dhead = x.shape
    dstate = B_mat.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    ngroups = B_mat.shape[2]

    assert seqlen <= nchunks * chunk_size
    assert nheads % ngroups == 0

    B_expanded = torch.repeat_interleave(B_mat, nheads // ngroups, dim=2)

    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B_expanded = F.pad(B_expanded, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))

    x = x.reshape(batch, nchunks, chunk_size, nheads, dhead)
    B_expanded = B_expanded.reshape(batch, nchunks, chunk_size, nheads, dstate)

    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))

    return {"output": torch.einsum(
        "bclhn,bhcl,bhcl,bclhp->bchpn",
        B_expanded.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x,
    )}
