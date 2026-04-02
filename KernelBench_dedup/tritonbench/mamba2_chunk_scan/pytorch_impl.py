import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cb: torch.Tensor, x: torch.Tensor, dt: torch.Tensor,
                dA_cumsum: torch.Tensor, C: torch.Tensor,
                prev_states: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """
        Eager Mamba2 chunk scan forward pass.

        Args:
            cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
            x: (batch, seqlen, nheads, dhead)
            dt: (batch, nheads, nchunks, chunk_size)
            dA_cumsum: (batch, nheads, nchunks, chunk_size)
            C: (batch, seqlen, ngroups, dstate)
            prev_states: (batch, nchunks, nheads, dhead, dstate)
            D: (nheads,)

        Returns:
            out: (batch, seqlen, nheads, dhead)
        """
        _, _, ngroups, _, _ = cb.shape
        batch, seqlen, nheads, dhead = x.shape
        _, _, nchunks, chunk_size = dt.shape
        dstate = C.shape[-1]

        assert seqlen == nchunks * chunk_size

        # Repeat interleave for groups
        C_expanded = torch.repeat_interleave(C, nheads // ngroups, dim=2)
        cb_expanded = torch.repeat_interleave(cb, nheads // ngroups, dim=2)

        # Compute decay
        dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
        decay = torch.exp(dt_segment_sum)
        scores_decay = cb_expanded * decay.permute(0, 2, 1, 3, 4)

        # Apply causal mask
        causal_mask = torch.tril(
            torch.ones(chunk_size, chunk_size, device=x.device, dtype=torch.bool),
            diagonal=0
        )
        scores_decay = scores_decay.masked_fill(~causal_mask, 0)

        # Intra-chunk attention
        out = torch.einsum(
            "bchls,bhcs,bcshp->bclhp",
            scores_decay.to(x.dtype),
            dt.to(x.dtype),
            x.reshape(batch, nchunks, chunk_size, nheads, dhead),
        )

        # Inter-chunk (state) contribution
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

        # D skip connection
        if D is not None:
            if D.dim() == 1:
                D = D.unsqueeze(-1)
            out = out + x * D

        return out


# Default shapes from operator.py
BATCH = 1
NHEADS = 64
NGROUPS = 1
SEQLEN = 1024
CHUNK_SIZE = 256
DHEAD = 64
DSTATE = 128
DTYPE = torch.float16


def get_inputs():
    torch.manual_seed(42)
    nchunks = (SEQLEN + CHUNK_SIZE - 1) // CHUNK_SIZE
    cb = torch.randn(BATCH, nchunks, NGROUPS, CHUNK_SIZE, CHUNK_SIZE, dtype=DTYPE)
    x = torch.randn(BATCH, SEQLEN, NHEADS, DHEAD, dtype=DTYPE)
    dt = torch.randn(BATCH, NHEADS, nchunks, CHUNK_SIZE, dtype=DTYPE)
    dA_cumsum = torch.rand(BATCH, NHEADS, nchunks, CHUNK_SIZE, dtype=DTYPE)
    C = torch.randn(BATCH, SEQLEN, NGROUPS, DSTATE, dtype=DTYPE)
    prev_states = torch.randn(BATCH, nchunks, NHEADS, DHEAD, DSTATE, dtype=DTYPE)
    D = torch.randn(NHEADS, dtype=DTYPE)
    return [cb, x, dt, dA_cumsum, C, prev_states, D]


def get_init_inputs():
    return []


def get_test_inputs():
    return [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
