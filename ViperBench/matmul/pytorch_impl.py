import torch


def matmul(a, b):
    """Unified API: matmul(a, b) -> Tensor for 2D float16 matrices."""
    return torch.matmul(a, b)
