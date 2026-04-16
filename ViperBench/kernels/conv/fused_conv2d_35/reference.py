"""
Reference implementation for: fused_conv2d_35
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
    Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = x - self.subtract_value
        x = torch.nn.functional.hardswish(x)
        x = self.pool(x)
        x = torch.nn.functional.mish(x)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]

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
