import torch
import torch.nn.functional as F

def relu(x):
    """Element-wise ReLU activation."""
    return F.relu(x)
