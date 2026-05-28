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
