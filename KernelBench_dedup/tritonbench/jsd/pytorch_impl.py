import torch
import torch.nn as nn


class Model(nn.Module):
    """Jensen-Shannon Divergence using PyTorch ops."""

    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="none", log_target=True)
        self.beta = beta

    def forward(self, log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
        log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
        log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(-1, log_q.size(-1))
        m = torch.lerp(torch.exp(log_q), torch.exp(log_p), self.beta)
        loss = self.beta * self.kl(torch.log(m), log_p).sum(dim=-1) + (
            1 - self.beta
        ) * self.kl(torch.log(m), log_q).sum(dim=-1)
        loss = (loss / log_q.shape[0]).sum()
        return loss


# Default shapes from operator.py: B=4, T=2048, first V=2^12=4096
B = 4
T = 2048
V = 4096


def get_inputs():
    # Both input and target are log-softmax distributions
    input_tensor = torch.randn(B * T, V, dtype=torch.float32).log_softmax(dim=-1)
    target = torch.randn(B * T, V, dtype=torch.float32).log_softmax(dim=-1)
    return [input_tensor, target]


def get_init_inputs():
    return [0.5]


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
