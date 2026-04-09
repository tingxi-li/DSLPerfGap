import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []


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
