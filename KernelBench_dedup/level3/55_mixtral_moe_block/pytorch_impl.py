import torch
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralConfig,
)


class Model(nn.Module):
    def __init__(self, hidden_size=4096, intermediate_size=14336,
                 num_attention_heads=32, num_key_value_heads=8,
                 num_local_experts=8, num_experts_per_tok=2,
                 max_position_embeddings=32768):
        super().__init__()
        config = MixtralConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=1e-5,
        )
        self.layer = MixtralDecoderLayer(config, layer_idx=0)

    def forward(self, hidden_states, position_ids):
        output = self.layer(hidden_states, position_ids=position_ids)
        return output[0]


# Mixtral-8x7B shapes
batch_size = 2
seq_len = 512
hidden_size = 4096
intermediate_size = 14336
num_attention_heads = 32
num_key_value_heads = 8
num_local_experts = 8
num_experts_per_tok = 2
max_position_embeddings = 32768


def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    return [hidden_states, position_ids]


def get_init_inputs():
    return [hidden_size, intermediate_size, num_attention_heads,
            num_key_value_heads, num_local_experts, num_experts_per_tok,
            max_position_embeddings]


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
