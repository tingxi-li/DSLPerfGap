import torch

def logsumexp(x):
    """Logsumexp reduction along the last dimension."""
    return torch.logsumexp(x, dim=-1)
