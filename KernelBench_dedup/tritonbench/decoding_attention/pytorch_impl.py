import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, batch: int, seq_len_q: int, seq_len_kv: int,
                 head_q: int, head_kv: int, head_d: int):
        super().__init__()
        self.batch = batch
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.head_q = head_q
        self.head_kv = head_kv
        self.head_d = head_d

    def forward(self, q: torch.Tensor, k_cache: torch.Tensor,
                v_cache: torch.Tensor, cache_seqlens: torch.Tensor) -> torch.Tensor:
        # q: (batch, seq_len_q, head_q, head_d)
        # k_cache, v_cache: (batch, max_len_kv, head_kv, head_d)
        # cache_seqlens: (batch,) -- actual kv lengths per batch element
        #
        # We use a simple loop over batch to handle variable-length kv caches,
        # applying SDPA per element. For the common decode case, seq_len_q=1.
        batch = q.shape[0]
        head_q = q.shape[2]
        head_kv = k_cache.shape[2]
        head_d = q.shape[3]

        # Transpose to (batch, heads, seq, d)
        q_t = q.transpose(1, 2)  # (B, Hq, Sq, D)

        # Expand kv heads for GQA: repeat each kv head for head_q // head_kv query heads
        heads_per_kv = head_q // head_kv
        k_t = k_cache.transpose(1, 2)  # (B, Hkv, Skv, D)
        v_t = v_cache.transpose(1, 2)  # (B, Hkv, Skv, D)
        if heads_per_kv > 1:
            k_t = k_t.repeat_interleave(heads_per_kv, dim=1)
            v_t = v_t.repeat_interleave(heads_per_kv, dim=1)

        # Process each batch element with its actual kv length
        outputs = []
        for i in range(batch):
            seq_kv = cache_seqlens[i].item()
            qi = q_t[i:i+1]              # (1, Hq, Sq, D)
            ki = k_t[i:i+1, :, :seq_kv]  # (1, Hq, seq_kv, D)
            vi = v_t[i:i+1, :, :seq_kv]  # (1, Hq, seq_kv, D)
            out_i = F.scaled_dot_product_attention(qi, ki, vi, is_causal=False)
            outputs.append(out_i)

        out = torch.cat(outputs, dim=0)  # (B, Hq, Sq, D)
        return out.transpose(1, 2)  # (B, Sq, Hq, D)


# Default shapes from operator.py
BATCH = 16
SEQ_LEN_Q = 1
SEQ_LEN_KV = 4096
MAX_LEN_KV = 8192
HEAD_Q = 8
HEAD_KV = 1
HEAD_D = 128
DTYPE = torch.bfloat16


def get_inputs():
    q = torch.randn(BATCH, SEQ_LEN_Q, HEAD_Q, HEAD_D, dtype=DTYPE)
    k_cache = torch.randn(BATCH, MAX_LEN_KV, HEAD_KV, HEAD_D, dtype=DTYPE)
    v_cache = torch.randn(BATCH, MAX_LEN_KV, HEAD_KV, HEAD_D, dtype=DTYPE)
    cache_seqlens = torch.tensor([SEQ_LEN_KV] * BATCH, dtype=torch.int32)
    return [q, k_cache, v_cache, cache_seqlens]


def get_init_inputs():
    return [BATCH, SEQ_LEN_Q, SEQ_LEN_KV, HEAD_Q, HEAD_KV, HEAD_D]


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
