import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Welford-based layer norm (eager F.layer_norm baseline)."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, weight: torch.Tensor, bias: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=(x.shape[-1],),
                            weight=weight, bias=bias, eps=self.eps)


# Default shapes from operator.py: first BUILDIN_SHAPE = (262144, 1024)
S = 262144
D = 1024


def get_inputs():
    # p1=weight, p2=bias, p3=input (matching operator.py argument order)
    weight = torch.randn(D, dtype=torch.bfloat16)
    bias = torch.randn(D, dtype=torch.bfloat16)
    x = torch.randn(S, D, dtype=torch.bfloat16)
    return [weight, bias, x]


def get_init_inputs():
    return [1e-5]


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
