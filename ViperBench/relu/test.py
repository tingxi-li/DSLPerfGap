import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import relu as pytorch_relu
from tilelang_impl import relu as tilelang_relu
from test_utils import run_tilelang_test

test_cases = [
    {"name": "small_f32", "dtype": torch.float32,
     "inputs": (torch.randn(16, device='cuda'),)},
    {"name": "medium_f32", "dtype": torch.float32,
     "inputs": (torch.randn(1024, device='cuda'),)},
    {"name": "large_f32", "dtype": torch.float32,
     "inputs": (torch.randn(100000, device='cuda'),)},
    {"name": "2d_f32", "dtype": torch.float32,
     "inputs": (torch.randn(128, 256, device='cuda'),)},
    {"name": "small_f16", "dtype": torch.float16,
     "inputs": (torch.randn(16, device='cuda', dtype=torch.float16),)},
    {"name": "large_f16", "dtype": torch.float16,
     "inputs": (torch.randn(100000, device='cuda', dtype=torch.float16),)},
    {"name": "negatives_f32", "dtype": torch.float32,
     "inputs": (torch.tensor([-3.0, -1.0, -0.5, -2.0, -5.0], device='cuda'),)},
    {"name": "mixed_f32", "dtype": torch.float32,
     "inputs": (torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0], device='cuda'),)},
]

if __name__ == '__main__':
    run_tilelang_test("relu", test_cases, pytorch_relu, tilelang_relu)
