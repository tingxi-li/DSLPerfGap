import torch
import torch.nn as nn


class Model(nn.Module):
    """Baseline gather + GEMV: w[idx].to(x.dtype) @ x."""

    def __init__(self):
        super().__init__()

    def forward(self, w, idx, x):
        return w[idx].to(x.dtype) @ x


# Default shape: S=2048 (2^11, the first from the operator's range(11,15))
S = 2048
NUM_EXPERTS = 8
NUM_SELECTED = 2
DTYPE = torch.bfloat16


def get_inputs():
    w = torch.randint(-128, 127, (NUM_EXPERTS, S, S), dtype=torch.int8)
    idx = torch.randint(0, NUM_EXPERTS, (NUM_SELECTED,), dtype=torch.int64)
    x = torch.randn(S, dtype=DTYPE)
    return [w, idx, x]


def get_init_inputs():
    return []


def get_test_inputs():
    w = torch.randint(-128, 127, (NUM_EXPERTS, S, S), dtype=torch.int8, device="cuda")
    idx = torch.randint(0, NUM_EXPERTS, (NUM_SELECTED,), dtype=torch.int64, device="cuda")
    x = torch.randn(S, dtype=DTYPE, device="cuda")
    return [w, idx, x]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
