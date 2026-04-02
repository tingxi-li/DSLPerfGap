import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        return x

batch_size = 1024
in_features  = 8192  
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]

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
