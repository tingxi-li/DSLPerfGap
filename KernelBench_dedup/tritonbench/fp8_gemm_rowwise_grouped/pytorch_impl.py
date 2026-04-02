import torch
import torch.nn as nn


class Model(nn.Module):
    """FP8 rowwise grouped GEMM fallback: float16 grouped matmul as a portable baseline.

    The real operator uses fp8 quantized tensors with row-wise scaling in a grouped
    GEMM pattern. This adapter uses float16 matmul per group and concatenates results.
    """

    def __init__(self):
        super().__init__()

    def forward(self, group_A, B_shared):
        # group_A: list of (M_i, K) tensors
        # B_shared: (K, N) shared weight
        outs = [torch.matmul(a, B_shared) for a in group_A]
        return torch.cat(outs, dim=0)


# Default shapes from BUILTIN_SHAPES and GROUP_SIZES
GROUP_SIZE = 4
M = 1024
K = 1024
N = 1024
DTYPE = torch.float16


def get_inputs():
    group_A = [torch.randn(M, K, dtype=DTYPE) for _ in range(GROUP_SIZE)]
    B_shared = torch.randn(K, N, dtype=DTYPE)
    return [group_A, B_shared]


def get_init_inputs():
    return []


def get_test_inputs():
    group_A = [torch.randn(M, K, dtype=DTYPE, device="cuda") for _ in range(GROUP_SIZE)]
    B_shared = torch.randn(K, N, dtype=DTYPE, device="cuda")
    return [group_A, B_shared]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
