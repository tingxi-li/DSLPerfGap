import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self) -> None:
        # No-op: this kernel measures launch latency only
        pass


def get_inputs():
    return []


def get_init_inputs():
    return []


def get_test_inputs():
    return []


def run(*args):
    model = Model().cuda().eval()
    with torch.no_grad():
        return model()
