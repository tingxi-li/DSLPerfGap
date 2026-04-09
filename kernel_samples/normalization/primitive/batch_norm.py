import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)

batch_size = 64
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]

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
