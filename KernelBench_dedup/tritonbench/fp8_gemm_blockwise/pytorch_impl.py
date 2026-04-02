import torch
import torch.nn as nn


class Model(nn.Module):
    """FP8 blockwise GEMM fallback: standard float16 matmul as a portable baseline.

    The real operator uses fp8 quantized tensors with block-wise scaling.
    This adapter uses float16 matmul to avoid fp8 hardware requirements.
    """

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        # a: (M, K), b: (K, N) in float16
        return torch.matmul(a, b)


# Default shape from BUILDIN_SHAPES
M = 2048
K = 8192
N = 2048
DTYPE = torch.float16


def get_inputs():
    a = torch.randn(M, K, dtype=DTYPE)
    b = torch.randn(K, N, dtype=DTYPE)
    return [a, b]


def get_init_inputs():
    return []


def get_test_inputs():
    return [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
