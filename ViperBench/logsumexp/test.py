import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import logsumexp as pytorch_logsumexp
from tilelang_impl import logsumexp as tilelang_logsumexp
from test_utils import run_tilelang_test

test_cases = [
    {"name": "small_2d", "dtype": torch.float32,
     "inputs": (torch.randn(4, 64, device='cuda'),)},
    {"name": "medium_2d", "dtype": torch.float32,
     "inputs": (torch.randn(32, 256, device='cuda'),)},
    {"name": "large_2d", "dtype": torch.float32,
     "inputs": (torch.randn(128, 1024, device='cuda'),)},
    {"name": "wide_2d", "dtype": torch.float32,
     "inputs": (torch.randn(8, 4096, device='cuda'),)},
    {"name": "3d_tensor", "dtype": torch.float32,
     "inputs": (torch.randn(4, 8, 64, device='cuda'),)},
    {"name": "4d_tensor", "dtype": torch.float32,
     "inputs": (torch.randn(2, 3, 4, 32, device='cuda'),)},
    {"name": "1d_tensor", "dtype": torch.float32,
     "inputs": (torch.randn(128, device='cuda'),)},
]

if __name__ == '__main__':
    run_tilelang_test("logsumexp", test_cases, pytorch_logsumexp, tilelang_logsumexp, loose_tol=True)
