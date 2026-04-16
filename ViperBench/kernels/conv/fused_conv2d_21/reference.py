"""
Reference implementation for: fused_conv2d_21
Source: KernelBench
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Original KernelBench source ──
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x + self.bias
        x = x * self.scale
        x = torch.sigmoid(x)
        x = self.group_norm(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]

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

# ── End original source ──

# ── ViperBench reference interface ──
_MODEL_CACHE = {}

def _get_model():
    key = "default"
    if key not in _MODEL_CACHE:
        init_args = get_init_inputs()
        model = Model(*init_args).cuda().eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model = _get_model()
    input_tensors = [inputs["input"]]
    with torch.no_grad():
        result = model(*input_tensors)
    if isinstance(result, tuple):
        return {"output_" + str(i): v for i, v in enumerate(result)}
    return {"output": result}
