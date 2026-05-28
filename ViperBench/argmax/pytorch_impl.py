import torch


def argmax(input_tensor, dim, keepdim=False):
    """
    Returns the indices of the maximum values across a specified dimension.
    """
    return torch.argmax(input_tensor, dim=dim, keepdim=keepdim)
