import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, reduction: str = "batchmean"):
        super().__init__()
        self.loss_fn = nn.KLDivLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(input, target)


# Default shapes from operator.py: B=8, T=512, first V=2^12=4096
B = 8
T = 512
V = 4096


def get_inputs():
    # input is log-softmax, target is softmax (probability distribution)
    input_tensor = torch.randn(B * T, V, dtype=torch.float32).log_softmax(dim=-1)
    target = torch.randn(B * T, V, dtype=torch.float32).softmax(dim=-1)
    return [input_tensor, target]


def get_init_inputs():
    return ["batchmean"]


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
