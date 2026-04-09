"""
Reference implementation for: model_siglip_encoder
Source: KernelBench
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Original KernelBench source ──
import torch
import torch.nn as nn
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionTransformer,
    SiglipVisionConfig,
)


class Model(nn.Module):
    def __init__(self, image_size=224, patch_size=16, hidden_size=768,
                 num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072):
        super().__init__()
        config = SiglipVisionConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            attn_implementation='sdpa',
        )
        self.encoder = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state


# SigLIP-Base shapes
batch_size = 4
image_size = 224
patch_size = 16
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
intermediate_size = 3072


def get_inputs():
    return [torch.randn(batch_size, 3, image_size, image_size)]


def get_init_inputs():
    return [image_size, patch_size, hidden_size,
            num_hidden_layers, num_attention_heads, intermediate_size]


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
