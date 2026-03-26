import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import softmax as pytorch_softmax
from triton_impl import softmax as triton_softmax
from tilelang_impl import softmax as tilelang_softmax
from test_utils import run_test, run_tilelang_test

test_cases = [
    {"name": "small_2d", "dtype": torch.float32,
     "inputs": (torch.randn(4, 8, device='cuda'),)},
    {"name": "medium_2d", "dtype": torch.float32,
     "inputs": (torch.randn(128, 512, device='cuda'),)},
    {"name": "large_cols", "dtype": torch.float32,
     "inputs": (torch.randn(64, 4096, device='cuda'),)},
    {"name": "wide", "dtype": torch.float32,
     "inputs": (torch.randn(16, 2048, device='cuda'),)},
    {"name": "square", "dtype": torch.float32,
     "inputs": (torch.randn(100, 100, device='cuda'),)},
    {"name": "3d_tensor", "dtype": torch.float32,
     "inputs": (torch.randn(4, 8, 64, device='cuda'),)},
    {"name": "4d_tensor", "dtype": torch.float32,
     "inputs": (torch.randn(2, 3, 4, 32, device='cuda'),)},
    {"name": "1d_tensor", "dtype": torch.float32,
     "inputs": (torch.randn(128, device='cuda'),)},
]

if __name__ == '__main__':
    run_tilelang_test("softmax", test_cases, pytorch_softmax, tilelang_softmax, loose_tol=True)
