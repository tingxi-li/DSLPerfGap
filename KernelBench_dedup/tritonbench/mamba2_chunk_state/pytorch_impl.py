import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, B_mat: torch.Tensor, x: torch.Tensor, dt: torch.Tensor,
                dA_cumsum: torch.Tensor) -> torch.Tensor:
        """
        Eager Mamba2 chunk state forward pass.

        Args:
            B_mat: (batch, seqlen, ngroups, dstate)
            x: (batch, seqlen, nheads, dhead)
            dt: (batch, nheads, nchunks, chunk_size)
            dA_cumsum: (batch, nheads, nchunks, chunk_size)

        Returns:
            states: (batch, nchunks, nheads, dhead, dstate)
        """
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

        return torch.einsum(
            "bclhn,bhcl,bhcl,bclhp->bchpn",
            B_expanded.to(x.dtype),
            decay_states.to(x.dtype),
            dt.to(x.dtype),
            x,
        )


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
    B_mat = torch.rand(BATCH, SEQLEN, NGROUPS, DSTATE, dtype=DTYPE)
    x = torch.rand(BATCH, SEQLEN, NHEADS, DHEAD, dtype=DTYPE)
    dt = torch.rand(BATCH, NHEADS, nchunks, CHUNK_SIZE, dtype=DTYPE)
    dA_cumsum = torch.rand(BATCH, NHEADS, nchunks, CHUNK_SIZE, dtype=DTYPE)
    return [B_mat, x, dt, dA_cumsum]


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
