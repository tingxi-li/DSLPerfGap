import torch
import torch.nn as nn
from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = resnet50(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


batch_size = 8
num_classes = 1000


def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]


def get_init_inputs():
    return [num_classes]


# ── Unified interface for eval harness ──────────────────────────────────────
def get_test_inputs():
    """Return ready-to-use CUDA inputs for testing."""
    return [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]


def run(*args):
    """Unified interface: instantiate Model, move to CUDA, run forward."""
    if args:
        inputs = args
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
