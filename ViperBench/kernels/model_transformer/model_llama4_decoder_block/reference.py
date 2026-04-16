"""
Reference implementation for: model_llama4_decoder_block
Source: KernelBench
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Original KernelBench source ──
import torch
import torch.nn as nn
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextDecoderLayer,
    Llama4TextConfig,
    Llama4TextRotaryEmbedding,
)


class Model(nn.Module):
    def __init__(self, hidden_size=5120, intermediate_size=8192,
                 num_attention_heads=40, num_key_value_heads=8,
                 num_local_experts=16, num_experts_per_tok=1,
                 max_position_embeddings=8192):
        super().__init__()
        self.config = Llama4TextConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_position_embeddings=max_position_embeddings,
            attn_implementation='sdpa',
            rms_norm_eps=1e-5,
            interleave_moe_layer_step=1,
            num_hidden_layers=2,
        )
        self.layer = Llama4TextDecoderLayer(self.config, layer_idx=0)
        self.rotary_emb = Llama4TextRotaryEmbedding(self.config)

    def forward(self, hidden_states, position_ids):
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        output = self.layer(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        return output[0]


batch_size = 2
seq_len = 512
hidden_size = 5120
intermediate_size = 8192
num_attention_heads = 40
num_key_value_heads = 8
num_local_experts = 16
num_experts_per_tok = 1
max_position_embeddings = 8192


def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    return [hidden_states, position_ids]


def get_init_inputs():
    return [hidden_size, intermediate_size, num_attention_heads,
            num_key_value_heads, num_local_experts, num_experts_per_tok,
            max_position_embeddings]


def get_test_inputs():
    return [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]


def run(*args):
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
