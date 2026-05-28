import torch
import torch.nn.functional as F

def log_softmax(x, dim=-1, dtype=None):
    """
    Log-softmax: y = log(softmax(x, dim))
    """
    return F.log_softmax(x, dim=dim, dtype=dtype)
