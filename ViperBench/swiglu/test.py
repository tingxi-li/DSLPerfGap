import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import swiglu as pytorch_swiglu
from triton_impl import swiglu as triton_swiglu
from tilelang_impl import swiglu as tilelang_swiglu
from test_utils import run_test, run_tilelang_test


def make_test(batch, ncols, dtype):
    return (torch.randn(batch, 2 * ncols, device='cuda', dtype=dtype),)


def pytorch_fn(xy):
    return pytorch_swiglu(xy)


def triton_fn(xy):
    return triton_swiglu(xy)


def tilelang_fn(xy):
    return tilelang_swiglu(xy)


if __name__ == "__main__":
    test_cases = [
        {"name": "small_fp32", "inputs": make_test(4, 64, torch.float32), "dtype": torch.float32},
        {"name": "medium_fp32", "inputs": make_test(32, 256, torch.float32), "dtype": torch.float32},
        {"name": "large_fp32", "inputs": make_test(128, 1024, torch.float32), "dtype": torch.float32},
        {"name": "edge_fp32", "inputs": make_test(1, 32, torch.float32), "dtype": torch.float32},
        {"name": "small_fp16", "inputs": make_test(16, 128, torch.float16), "dtype": torch.float16},
    ]
    run_tilelang_test("swiglu", test_cases, pytorch_fn, tilelang_fn)
