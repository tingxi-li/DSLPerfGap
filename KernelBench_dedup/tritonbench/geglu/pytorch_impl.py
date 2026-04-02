import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """GEGLU MLP: down_proj(gelu(gate_proj(x)) * up_proj(x))
    Matches LlamaMLP with hidden_act='gelu_pytorch_tanh'."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


# Default shapes from operator.py: bsz=8, first T=1024 (2^10), hidden=4096, intermediate=11008
BSZ = 8
SEQ_LEN = 1024
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 11008


def get_inputs():
    return [torch.randn(BSZ, SEQ_LEN, HIDDEN_SIZE, dtype=torch.float32)]


def get_init_inputs():
    return [HIDDEN_SIZE, INTERMEDIATE_SIZE]


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
