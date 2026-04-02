import torch
import torch.nn as nn


class Model(nn.Module):
    """FP8 GEMM fallback: standard float16 matmul as a safe portable baseline.

    The real operator uses torch._scaled_mm with fp8 tensors and scaling factors.
    This adapter uses float16 matmul to avoid fp8 hardware requirements.
    """

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        # a: (M, K), b: (N, K) stored as fp8 in real op, but here we use fp16
        return torch.matmul(a, b.t())


# Default shape: M=4096, K=4096, N=4096
M = 4096
K = 4096
N = 4096
DTYPE = torch.float16


def get_inputs():
    a = torch.randn(M, K, dtype=DTYPE)
    b = torch.randn(N, K, dtype=DTYPE)
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
