import torch
import torch.nn as nn
import math


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, dim]
    sin = sin.unsqueeze(1)  # [batch, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Model(nn.Module):
    def __init__(self, head_dim: int, num_q_heads: int, num_kv_heads: int,
                 seq_length: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.seq_length = seq_length
        # Precompute cos/sin for rotary embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        # Compute cos/sin
        t = torch.arange(self.seq_length, device=q.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(q.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0)  # [1, seq_len, dim]
        sin = emb.sin().unsqueeze(0)  # [1, seq_len, dim]
        return apply_rotary_pos_emb(q, k, cos, sin)


# Default shapes from operator.py: hidden_size=8192, seq_length=1024
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HIDDEN_SIZE = 8192
HEAD_DIM = HIDDEN_SIZE // NUM_Q_HEADS  # 256
SEQ_LENGTH = 1024


def get_inputs():
    # q: (1, num_q_heads, seq_length, head_dim)
    # k: (1, num_kv_heads, seq_length, head_dim)
    q = torch.randn(1, NUM_Q_HEADS, SEQ_LENGTH, HEAD_DIM, dtype=torch.float32)
    k = torch.randn(1, NUM_KV_HEADS, SEQ_LENGTH, HEAD_DIM, dtype=torch.float32)
    return [q, k]


def get_init_inputs():
    return [HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, SEQ_LENGTH]


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
