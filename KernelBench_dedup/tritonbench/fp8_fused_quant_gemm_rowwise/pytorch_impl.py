import torch
import torch.nn as nn


class Model(nn.Module):
    """FP8 fused quantization + GEMM rowwise fallback: float16 matmul baseline.

    The real operator fuses RMS norm or SiLU+mul with fp8 row-wise quantization
    and then performs fp8 GEMM. This adapter simply performs the matmul in float16
    as a portable baseline (x1 @ w^T).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x1, w):
        # x1: (M, K), w: (N, K) -- matmul with transposed weight
        return torch.matmul(x1, w.t())


# Default shape from BUILDIN_SHAPES
M = 2048
K = 8192
N = 2048
DTYPE = torch.float16


def get_inputs():
    x1 = torch.randn(M, K, dtype=DTYPE)
    w = torch.randn(N, K, dtype=DTYPE)
    return [x1, w]


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
