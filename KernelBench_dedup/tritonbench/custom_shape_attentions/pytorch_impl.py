import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: (batch, n_heads, seq_len_q, d_head)
        # k, v: (batch, n_heads_kv, seq_len_kv, d_head)
        d_head = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(d_head)
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, scale=sm_scale
        )


# Default shapes: customizable attention shapes
BATCH = 4
N_HEADS = 48
N_HEADS_KV = 48
SEQ_LEN = 1024
SEQ_LEN_KV = 1024
D_HEAD = 128
CAUSAL = True
DTYPE = torch.bfloat16


def get_inputs():
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, dtype=DTYPE)
    k = torch.randn(BATCH, N_HEADS_KV, SEQ_LEN_KV, D_HEAD, dtype=DTYPE)
    v = torch.randn(BATCH, N_HEADS_KV, SEQ_LEN_KV, D_HEAD, dtype=DTYPE)
    return [q, k, v]


def get_init_inputs():
    return [CAUSAL]


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
