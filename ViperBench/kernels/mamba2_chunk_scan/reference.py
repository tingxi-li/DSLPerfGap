"""Reference: mamba2_chunk_scan — Mamba2 chunk scan forward pass."""
import torch

def reference(inputs):
    cb = inputs["cb"]
    x = inputs["x"]
    dt = inputs["dt"]
    dA_cumsum = inputs["dA_cumsum"]
    C = inputs["C"]
    prev_states = inputs["prev_states"]
    D = inputs["D"]

    _, _, ngroups, _, _ = cb.shape
    batch, seqlen, nheads, dhead = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = C.shape[-1]

    assert seqlen == nchunks * chunk_size

    C_expanded = torch.repeat_interleave(C, nheads // ngroups, dim=2)
    cb_expanded = torch.repeat_interleave(cb, nheads // ngroups, dim=2)

    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = cb_expanded * decay.permute(0, 2, 1, 3, 4)

    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=x.device, dtype=torch.bool), diagonal=0
    )
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)

    out = torch.einsum(
        "bchls,bhcs,bcshp->bclhp",
        scores_decay.to(x.dtype), dt.to(x.dtype),
        x.reshape(batch, nchunks, chunk_size, nheads, dhead),
    )

    state_decay_out = torch.exp(dA_cumsum.permute(0, 2, 3, 1).unsqueeze(-1))
    out_prev = (
        torch.einsum(
            "bclhn,bchpn->bclhp",
            C_expanded.reshape(batch, nchunks, chunk_size, nheads, dstate),
            prev_states.to(C_expanded.dtype),
        )
        * state_decay_out
    )
    out = out + out_prev
    out = out.reshape(batch, seqlen, nheads, dhead)

    if D is not None:
        if D.dim() == 1:
            D = D.unsqueeze(-1)
        out = out + x * D

    return {"output": out}
