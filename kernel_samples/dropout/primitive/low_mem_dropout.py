import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, p: float = 0.25):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manual dropout: generate mask, apply mask and scale
        # Using manual implementation to match the tritonbench baseline
        mask = (torch.rand_like(x) > self.p).to(x.dtype)
        return x * mask / (1.0 - self.p)


# Default shapes
P = 0.25
SIZE = 2**18  # 262144


def get_inputs():
    return [torch.randn(SIZE, dtype=torch.float32)]


def get_init_inputs():
    return [P]


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
