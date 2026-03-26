import torch

def rms_norm(x, normalized_shape, weight, eps=1e-5):
    """
    RMS normalization.
    y = x / sqrt(mean(x^2) + eps) * weight
    """
    dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=dims, keepdim=True) + eps)
    return ((x.float() / rms) * weight.float()).to(x.dtype)
