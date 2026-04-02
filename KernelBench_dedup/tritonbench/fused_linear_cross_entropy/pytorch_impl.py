import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Fused linear + cross entropy baseline: F.linear then CrossEntropyLoss.

    The real operator fuses the linear projection and cross entropy loss
    computation. This adapter performs them sequentially as the PyTorch baseline.
    """

    def __init__(self, hidden_size=4096, vocab_size=128256, ignore_index=-100):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def forward(self, input, weight, target):
        logits = F.linear(input, weight)
        return self.ce_loss(logits, target)


# Default shapes
HIDDEN_SIZE = 4096
VOCAB_SIZE = 128256
BT = 4096  # batch*time dimension (2^12)
DTYPE = torch.float32


def get_inputs():
    input = torch.randn(BT, HIDDEN_SIZE, dtype=DTYPE)
    weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, dtype=DTYPE)
    target = torch.randint(0, VOCAB_SIZE, (BT,), dtype=torch.long)
    return [input, weight, target]


def get_init_inputs():
    return [HIDDEN_SIZE, VOCAB_SIZE]


def get_test_inputs():
    inputs = get_inputs()
    return [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
