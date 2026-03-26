import torch
import torch.nn.functional as F


def layer_norm(x, weight, bias, eps=1e-5):
    """Unified API: layer_norm(x, weight, bias, eps) -> Tensor
    Note: eps is fixed at 1e-5 to match the Triton kernel.
    """
    if eps != 1e-5:
        raise ValueError(f"Only eps=1e-5 is supported (got {eps}). "
                         "The Triton kernel hardcodes eps=1e-5.")
    return F.layer_norm(x, weight.shape, weight, bias, eps)
