import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from test_utils import run_tilelang_test
from pytorch_impl import mul as pytorch_mul
from tilelang_impl import mul as tilelang_mul
import torch

test_cases = [
    {"name": "small_1d", "inputs": (torch.randn(128, device='cuda', dtype=torch.float32),), "dtype": torch.float32},
    {"name": "medium_1d", "inputs": (torch.randn(1024*1024, device='cuda', dtype=torch.float32),), "dtype": torch.float32},
    {"name": "2d", "inputs": (torch.randn(256, 256, device='cuda', dtype=torch.float32),), "dtype": torch.float32},
    {"name": "fp16", "inputs": (torch.randn(4096, device='cuda', dtype=torch.float16),), "dtype": torch.float16},
]

run_tilelang_test("mul", test_cases, pytorch_mul, tilelang_mul)
