import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = torch.sum(x, dim=1, keepdim=True) 
        return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]

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
