import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.

        Args:
            dim (int): Dimension along which to normalize.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, dim, *).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return x / torch.norm(x, p=2, dim=1, keepdim=True)

batch_size = 32768
# choose dim so total <2^31
dim = 65535

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

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
