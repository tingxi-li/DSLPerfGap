import torch

def swiglu(xy, out=None):
    """
    SwiGLU forward pass.
    Splits input along last dim into x, y; computes x * sigmoid(x) * y.
    """
    x, y = xy.chunk(2, dim=-1)
    result = x * torch.sigmoid(x) * y
    if out is not None:
        out.copy_(result)
        return out
    return result
