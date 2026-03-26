import torch


def max_reduction(input, dim, keepdim=False):
    """
    Computes max along a specified dimension, returning (values, indices).
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError('The input must be a torch.Tensor.')
    values, indices = torch.max(input, dim, keepdim=keepdim)
    return (values, indices)
