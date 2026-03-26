import torch
import torch.nn.functional as F

def softmax(x):
    """Softmax along the last dimension for any shape tensor."""
    return F.softmax(x, dim=-1)
