import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, activation: str = "fast_gelu"):
        super().__init__()
        self.activation = activation

    def forward(self, jagged_q: torch.Tensor, jagged_k: torch.Tensor,
                jagged_v: torch.Tensor, q_offsets: torch.Tensor,
                k_offsets: torch.Tensor) -> torch.Tensor:
        """
        Generalized Dot Product Attention (GDPA) with activation function.

        Args:
            jagged_q: (total_q_tokens, H, D) -- jagged query
            jagged_k: (total_k_tokens, H, D) -- jagged key
            jagged_v: (total_k_tokens, H, D) -- jagged value
            q_offsets: (B+1,) cumulative offsets for queries
            k_offsets: (B+1,) cumulative offsets for keys/values

        Returns:
            output: (total_q_tokens, H, D)
        """
        B = q_offsets.shape[0] - 1
        outputs = []

        for i in range(B):
            q_start = q_offsets[i].item()
            q_end = q_offsets[i + 1].item()
            k_start = k_offsets[i].item()
            k_end = k_offsets[i + 1].item()

            # (seq_q, H, D) -> (1, H, seq_q, D)
            q_seq = jagged_q[q_start:q_end].transpose(0, 1).unsqueeze(0)
            k_seq = jagged_k[k_start:k_end].transpose(0, 1).unsqueeze(0)
            v_seq = jagged_v[k_start:k_end].transpose(0, 1).unsqueeze(0)

            # Compute attention scores
            scores = torch.matmul(q_seq, k_seq.transpose(-2, -1))  # (1, H, sq, sk)

            # Apply activation instead of softmax
            if self.activation == "gelu":
                scores = F.gelu(scores)
            elif self.activation == "fast_gelu":
                scores = F.gelu(scores, approximate="tanh")
            elif self.activation == "tanh":
                scores = torch.tanh(scores)

            # Compute output
            output = torch.matmul(scores, v_seq)  # (1, H, sq, D)
            # (1, H, sq, D) -> (sq, H, D)
            outputs.append(output.squeeze(0).transpose(0, 1))

        return torch.cat(outputs, dim=0)


# Default shapes from operator.py
BATCH = 1024
MAX_SEQ_LEN = 1000
DIM = 512
HEAD = 4
SPARSITY = 0.5
ACTIVATION = "fast_gelu"
DTYPE = torch.bfloat16


def get_inputs():
    # Create simple uniform-length jagged data for testing
    head_dim = DIM // HEAD
    seq_len = max(1, int(MAX_SEQ_LEN * (1.0 - SPARSITY)))
    total_tokens = BATCH * seq_len

    jagged_q = torch.randn(total_tokens, HEAD, head_dim, dtype=DTYPE)
    jagged_k = torch.randn(total_tokens, HEAD, head_dim, dtype=DTYPE)
    jagged_v = torch.randn(total_tokens, HEAD, head_dim, dtype=DTYPE)
    q_offsets = torch.arange(0, (BATCH + 1) * seq_len, seq_len, dtype=torch.int32)
    k_offsets = q_offsets.clone()
    return [jagged_q, jagged_k, jagged_v, q_offsets, k_offsets]


def get_init_inputs():
    return [ACTIVATION]


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
