import torch
import torch.nn as nn

# --- Original implementation inlined below ---
import torch
import torch.nn.functional as F


def conv2d(input, weight, bias=None, stride=1, padding=0, groups=1):
    """
    Applies a 2D convolution over an input image.
    Note: dilation is not supported (always 1).
    """
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    return F.conv2d(input, weight, bias, stride, padding, (1, 1), groups)

# --- End original implementation ---


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return conv2d(*args)


def get_inputs():
    x = torch.randn(32, 256, 128, 128, device='cuda', dtype=torch.float16)
    w = torch.randn(256, 256, 3, 3, device='cuda', dtype=torch.float16)
    return [x, w]

def get_init_inputs():
    return []
