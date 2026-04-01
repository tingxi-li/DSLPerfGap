import torch
import torch.nn as nn

# --- Original implementation inlined below ---
import torch


def matmul(a, b):
    """Unified API: matmul(a, b) -> Tensor for 2D float16 matrices."""
    return torch.matmul(a, b)

# --- End original implementation ---


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return matmul(*args)


def get_inputs():
    a = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    return [a, b]

def get_init_inputs():
    return []
