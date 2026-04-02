import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.matmul(x)
        x = x * self.scale_factor
        x = x + x
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * torch.nn.functional.mish(x)  # Mish activation
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]

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
