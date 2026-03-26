import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import rms_norm as pytorch_rms_norm
from triton_impl import rms_norm as triton_rms_norm
from tilelang_impl import rms_norm as tilelang_rms_norm
from test_utils import run_test, run_tilelang_test


def make_test(batch, features, dtype):
    x = torch.randn(batch, features, device='cuda', dtype=dtype)
    weight = torch.randn(features, device='cuda', dtype=dtype)
    return x, (features,), weight


def pytorch_fn(x, normalized_shape, weight):
    return pytorch_rms_norm(x, normalized_shape, weight)


def triton_fn(x, normalized_shape, weight):
    return triton_rms_norm(x, normalized_shape, weight)


def tilelang_fn(x, normalized_shape, weight):
    return tilelang_rms_norm(x, normalized_shape, weight)


if __name__ == "__main__":
    test_cases = [
        {"name": "small_fp32", "inputs": make_test(4, 64, torch.float32), "dtype": torch.float32},
        {"name": "medium_fp32", "inputs": make_test(32, 128, torch.float32), "dtype": torch.float32},
        {"name": "large_fp32", "inputs": make_test(128, 512, torch.float32), "dtype": torch.float32},
        {"name": "edge_fp32", "inputs": make_test(1, 32, torch.float32), "dtype": torch.float32},
        {"name": "small_fp16", "inputs": make_test(32, 128, torch.float16), "dtype": torch.float16},
    ]
    run_tilelang_test("rms_norm", test_cases, pytorch_fn, tilelang_fn, loose_tol=True)
