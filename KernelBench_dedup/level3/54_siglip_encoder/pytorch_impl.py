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
