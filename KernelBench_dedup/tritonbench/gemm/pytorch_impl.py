import torch
import torch.nn as nn


class Model(nn.Module):
    """Baseline GEMM: torch.matmul(a, b) with optional bias add."""

    def __init__(self, use_bias=False, bias_size=None):
        super().__init__()
        self.use_bias = use_bias
        self.bias_size = bias_size

    def forward(self, a, b, bias=None):
        out = torch.matmul(a, b)
        if bias is not None:
            out = out + bias
        return out


# Default shapes from BUILDIN_SHAPES: (M, N, K, Bias)
M = 8192
N = 8192
K = 4096
DTYPE = torch.float16


def get_inputs():
    a = torch.randn(M, K, dtype=DTYPE)
    b = torch.randn(K, N, dtype=DTYPE)
    bias = None
    return [a, b, bias]


def get_init_inputs():
    return []


def get_test_inputs():
    inputs = get_inputs()
    return [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
