import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import log_softmax as pytorch_log_softmax
from triton_impl import log_softmax as triton_log_softmax
from tilelang_impl import log_softmax as tilelang_log_softmax
from test_utils import run_test, run_tilelang_test


def pytorch_fn(x):
    return pytorch_log_softmax(x, dim=-1)


def triton_fn(x):
    return triton_log_softmax(x, dim=-1)


def tilelang_fn(x):
    return tilelang_log_softmax(x, dim=-1)


if __name__ == "__main__":
    test_cases = [
        {"name": "small_fp32", "inputs": (torch.randn(4, 64, device='cuda', dtype=torch.float32),), "dtype": torch.float32},
        {"name": "medium_fp32", "inputs": (torch.randn(32, 256, device='cuda', dtype=torch.float32),), "dtype": torch.float32},
        {"name": "large_fp32", "inputs": (torch.randn(128, 1024, device='cuda', dtype=torch.float32),), "dtype": torch.float32},
        {"name": "4d_fp32", "inputs": (torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float32),), "dtype": torch.float32},
        {"name": "small_fp16", "inputs": (torch.randn(32, 256, device='cuda', dtype=torch.float16),), "dtype": torch.float16},
    ]
    run_tilelang_test("log_softmax", test_cases, pytorch_fn, tilelang_fn, loose_tol=True)
