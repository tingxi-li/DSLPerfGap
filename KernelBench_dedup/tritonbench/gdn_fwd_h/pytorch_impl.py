import math

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, k: torch.Tensor, w: torch.Tensor, u: torch.Tensor,
                g: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """
        Eager gated delta network forward pass for computing h states.

        Args:
            k: (batch, seqlen, nheads, dhead)
            w: (batch, seqlen, nheads, dhead)
            u: (batch, seqlen, nheads, expand_v*dhead)
            g: (batch, seqlen, nheads)
            chunk_size: int

        Returns:
            h: (batch, nchunks, nheads, dhead, expand_v*dhead)
        """
        batch, seqlen, nheads, dhead = k.shape
        expand_v = u.shape[-1] // dhead
        nchunks = (seqlen + chunk_size - 1) // chunk_size

        acc_dtype = torch.float32
        dtype = k.dtype

        h = torch.empty(
            batch, nchunks, nheads, dhead, expand_v * dhead,
            dtype=dtype, device=k.device
        )
        b_h = torch.zeros(
            batch, nheads, dhead, expand_v * dhead,
            dtype=acc_dtype, device=k.device
        )

        k_c = k.reshape(batch, nchunks, chunk_size, nheads, dhead)
        w_c = w.reshape(batch, nchunks, chunk_size, nheads, dhead)
        u_c = u.reshape(batch, nchunks, chunk_size, nheads, expand_v * dhead)
        g_c = g.reshape(batch, nchunks, chunk_size, nheads)

        for i_t in range(nchunks):
            h[:, i_t, :, :, :] = b_h.to(dtype)
            b_w = w_c[:, i_t, :, :, :].to(acc_dtype)
            c_h = b_h.to(dtype).to(acc_dtype)
            b_v = torch.einsum("bchk,bhkv->bchv", b_w, c_h)
            p_v = u_c[:, i_t, :, :, :].to(acc_dtype)
            b_v = p_v - b_v
            last_idx = min((i_t + 1) * chunk_size, seqlen) - 1
            m_t = (i_t * chunk_size + torch.arange(0, chunk_size, device=k.device)) < seqlen
            b_g_last = g[:, last_idx, :].to(acc_dtype)
            b_g = g_c[:, i_t, :, :].to(acc_dtype)
            b_v *= torch.where(
                m_t.unsqueeze(0).unsqueeze(-1),
                torch.exp(b_g_last.unsqueeze(1) - b_g), 0
            ).unsqueeze(-1)
            b_g_last = torch.exp(b_g_last)
            b_h *= b_g_last.unsqueeze(-1).unsqueeze(-1)
            b_v = b_v.to(dtype).to(acc_dtype)
            p_k = k_c[:, i_t, :, :, :].to(acc_dtype)
            b_h += torch.einsum("bchk,bchv->bhkv", p_k, b_v)

        return h


# Default shapes from operator.py
BATCH = 1
NHEADS = 6
SEQLEN = 1024
CHUNK_SIZE = 64
DHEAD = 256
EXPAND_V = 2
DTYPE = torch.bfloat16


def get_inputs():
    torch.manual_seed(42)
    nchunks = (SEQLEN + CHUNK_SIZE - 1) // CHUNK_SIZE
    k = torch.randn(BATCH, SEQLEN, NHEADS, DHEAD, dtype=DTYPE)
    # w needs SVD processing like in operator.py for stability,
    # but for a simple baseline we just use randn
    w = torch.randn(BATCH, SEQLEN, NHEADS, DHEAD, dtype=DTYPE)
    u = torch.randn(BATCH, SEQLEN, NHEADS, EXPAND_V * DHEAD, dtype=DTYPE)
    g = torch.cumsum(
        0.5 * math.log(1 / DHEAD)
        * torch.rand(BATCH, SEQLEN, NHEADS, dtype=torch.float32),
        dim=1,
    )
    return [k, w, u, g, CHUNK_SIZE]


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
