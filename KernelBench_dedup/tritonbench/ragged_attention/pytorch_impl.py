import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, batch_size: int, num_heads: int, attn_dim: int, hidden_dim: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.hidden_dim = hidden_dim
        self.alpha = 1.0 / attn_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                seq_offsets: torch.Tensor) -> torch.Tensor:
        # q: (total_tokens, num_heads, attn_dim)
        # k: (total_tokens, num_heads, attn_dim)
        # v: (total_tokens, num_heads, hidden_dim)
        # seq_offsets: (batch+1,) cumulative offsets
        #
        # Process each sequence independently using batched SDPA
        B = seq_offsets.shape[0] - 1
        outputs = []
        for i in range(B):
            start = seq_offsets[i].item()
            end = seq_offsets[i + 1].item()
            seq_len = end - start

            # Extract per-sequence tensors: (seq_len, H, D) -> (1, H, seq_len, D)
            qi = q[start:end].transpose(0, 1).unsqueeze(0)
            ki = k[start:end].transpose(0, 1).unsqueeze(0)
            vi = v[start:end].transpose(0, 1).unsqueeze(0)

            # Compute causal attention with lower-triangular mask
            out_i = F.scaled_dot_product_attention(
                qi, ki, vi, is_causal=True, scale=self.alpha
            )
            # (1, H, seq_len, D) -> (seq_len, H, D)
            outputs.append(out_i.squeeze(0).transpose(0, 1))

        return torch.cat(outputs, dim=0)


# Default shapes from operator.py
BATCH_SIZE = 128
NUM_HEADS = 4
ATTN_DIM = 128
HIDDEN_DIM = 128
SEQ_LEN = 256  # per-sequence length


def get_inputs():
    total_tokens = BATCH_SIZE * SEQ_LEN
    q = torch.randn(total_tokens, NUM_HEADS, ATTN_DIM, dtype=torch.bfloat16)
    k = torch.randn(total_tokens, NUM_HEADS, ATTN_DIM, dtype=torch.bfloat16)
    v = torch.randn(total_tokens, NUM_HEADS, HIDDEN_DIM, dtype=torch.bfloat16)
    seq_offsets = torch.arange(0, (BATCH_SIZE + 1) * SEQ_LEN, SEQ_LEN, dtype=torch.int64)
    return [q, k, v, seq_offsets]


def get_init_inputs():
    return [BATCH_SIZE, NUM_HEADS, ATTN_DIM, HIDDEN_DIM]


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
