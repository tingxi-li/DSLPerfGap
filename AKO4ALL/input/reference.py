import torch
import torch.nn as nn

# --- Original implementation inlined below ---
import torch
import torch.nn.functional as F

def softmax(x):
    """Softmax along the last dimension for any shape tensor."""
    return F.softmax(x, dim=-1)

# --- End original implementation ---


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return softmax(*args)


def get_inputs():
    return [torch.randn(4096, 32768, device='cuda', dtype=torch.float16)]

def get_init_inputs():
    return []
