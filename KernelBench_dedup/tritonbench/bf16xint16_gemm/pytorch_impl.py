import torch
import torch.nn as nn


class Model(nn.Module):
    """BF16 x INT16 GEMM baseline: cast int16 weights to bf16 and matmul.

    The real operator performs a bf16 activation x int16 weight GEMM using
    a Triton kernel. This adapter casts int16 weights to bf16 and does
    a standard matmul.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        # x: (M, K) bf16, w: (K, N) int16
        w_bf16 = w.to(torch.bfloat16)
        return torch.matmul(x, w_bf16)


# Default shape: first yielded from get_input_iter (2^16, 1280, 8192)
M = 65536
DIN = 8192
DOUT = 1280
DTYPE = torch.bfloat16


def get_inputs():
    x = torch.randn(M, DIN, dtype=DTYPE)
    w = torch.randint(-(2**15), 2**15 - 1, (DIN, DOUT), dtype=torch.int16)
    return [x, w]


def get_init_inputs():
    return []


def get_test_inputs():
    x = torch.randn(M, DIN, dtype=DTYPE, device="cuda")
    w = torch.randint(-(2**15), 2**15 - 1, (DIN, DOUT), dtype=torch.int16, device="cuda")
    return [x, w]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
