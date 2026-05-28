import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import embedding as pytorch_embedding
from triton_impl import embedding as triton_embedding
from test_utils import run_test, run_tilelang_test
from tilelang_impl import embedding as tilelang_embedding


def make_test(vocab_size, embed_dim, seq_len, vob_start, vob_end):
    input_ids = torch.randint(vob_start, vob_end, (seq_len,), dtype=torch.int32, device='cuda')
    weight = torch.randn(vocab_size, embed_dim, dtype=torch.float32, device='cuda')
    return input_ids, weight, vob_start, vob_end


def pytorch_fn(input_ids, weight, vob_start, vob_end):
    out = torch.zeros(input_ids.shape[0], weight.shape[1], dtype=torch.float32, device='cuda')
    return pytorch_embedding(input_ids, weight, vob_start, vob_end, out)


def triton_fn(input_ids, weight, vob_start, vob_end):
    out = torch.zeros(input_ids.shape[0], weight.shape[1], dtype=torch.float32, device='cuda')
    triton_embedding(input_ids, weight, vob_start, vob_end, out)
    return out


if __name__ == "__main__":
    test_cases = [
        {"name": "small", "inputs": make_test(100, 64, 16, 0, 100), "dtype": torch.float32},
        {"name": "medium", "inputs": make_test(1000, 512, 128, 10, 1000), "dtype": torch.float32},
        {"name": "large", "inputs": make_test(5000, 256, 512, 0, 5000), "dtype": torch.float32},
        {"name": "partial_vocab", "inputs": make_test(1000, 128, 64, 100, 500), "dtype": torch.float32},
    ]
    def tilelang_fn(input_ids, weight, vob_start, vob_end):
        out = torch.zeros(input_ids.shape[0], weight.shape[1], dtype=torch.float32, device='cuda')
        return tilelang_embedding(input_ids, weight, vob_start, vob_end, out)

    run_tilelang_test("embedding", test_cases, pytorch_fn, tilelang_fn)
