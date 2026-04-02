import torch
import torch.nn as nn


class Model(nn.Module):
    """Baseline addmm: torch.addmm(a, mat1, mat2)."""

    def __init__(self):
        super().__init__()

    def forward(self, a, mat1, mat2):
        return torch.addmm(a, mat1, mat2)


# Default shape from BUILDIN_SHAPES: (M, K, N, BIAS_1D_Y)
M = 20120
K = 1536
N = 512
DTYPE = torch.float16


def get_inputs():
    a = torch.randn(M, N, dtype=DTYPE)
    mat1 = torch.randn(M, K, dtype=DTYPE)
    mat2 = torch.randn(K, N, dtype=DTYPE)
    return [a, mat1, mat2]


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
