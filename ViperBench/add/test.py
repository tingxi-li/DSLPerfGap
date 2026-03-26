import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import add as pytorch_add
from tilelang_impl import add as tilelang_add
from test_utils import run_tilelang_test

test_cases = [
    {"name": "small_f32", "dtype": torch.float32,
     "inputs": (torch.randn(16, device='cuda'), torch.randn(16, device='cuda'))},
    {"name": "medium_f32", "dtype": torch.float32,
     "inputs": (torch.randn(1024, device='cuda'), torch.randn(1024, device='cuda'))},
    {"name": "large_f32", "dtype": torch.float32,
     "inputs": (torch.randn(1000000, device='cuda'), torch.randn(1000000, device='cuda'))},
    {"name": "2d_f32", "dtype": torch.float32,
     "inputs": (torch.randn(128, 256, device='cuda'), torch.randn(128, 256, device='cuda'))},
    {"name": "small_f16", "dtype": torch.float16,
     "inputs": (torch.randn(16, device='cuda', dtype=torch.float16),
                torch.randn(16, device='cuda', dtype=torch.float16))},
    {"name": "large_f16", "dtype": torch.float16,
     "inputs": (torch.randn(100000, device='cuda', dtype=torch.float16),
                torch.randn(100000, device='cuda', dtype=torch.float16))},
]

if __name__ == '__main__':
    run_tilelang_test("add", test_cases, pytorch_add, tilelang_add)
