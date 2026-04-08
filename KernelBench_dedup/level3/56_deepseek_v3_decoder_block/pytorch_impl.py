import torch
import torch.nn as nn
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3Config,
    DeepseekV3RotaryEmbedding,
)


class Model(nn.Module):
    def __init__(self, hidden_size=2048, intermediate_size=5632,
                 num_attention_heads=16, num_key_value_heads=16,
                 n_routed_experts=16, n_shared_experts=1,
                 num_experts_per_tok=4,
                 qk_nope_head_dim=64, qk_rope_head_dim=32,
                 v_head_dim=64, kv_lora_rank=256,
                 max_position_embeddings=4096):
        super().__init__()
        self.config = DeepseekV3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            num_experts_per_tok=num_experts_per_tok,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            attn_implementation='sdpa',
            rms_norm_eps=1e-6,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            num_hidden_layers=2,
        )
        self.layer = DeepseekV3DecoderLayer(self.config, layer_idx=1)
        self.rotary_emb = DeepseekV3RotaryEmbedding(self.config)

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
hidden_size = 2048
intermediate_size = 5632
num_attention_heads = 16
num_key_value_heads = 16
n_routed_experts = 16
n_shared_experts = 1
num_experts_per_tok = 4
qk_nope_head_dim = 64
qk_rope_head_dim = 32
v_head_dim = 64
kv_lora_rank = 256
max_position_embeddings = 4096


def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    return [hidden_states, position_ids]


def get_init_inputs():
    return [hidden_size, intermediate_size, num_attention_heads,
            num_key_value_heads, n_routed_experts, n_shared_experts,
            num_experts_per_tok, qk_nope_head_dim, qk_rope_head_dim,
            v_head_dim, kv_lora_rank, max_position_embeddings]


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
