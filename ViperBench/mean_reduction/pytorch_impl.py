import torch


def mean_reduction(input_tensor, dim, keepdim=False, dtype=None):
    """
    Computes the mean value along specified dimensions.
    """
    return torch.mean(input_tensor, dim, keepdim=keepdim, dtype=dtype)
