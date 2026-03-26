import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from test_utils import run_tilelang_test
from pytorch_impl import batched_matmul as pytorch_fn
from tilelang_impl import batched_matmul as tilelang_batched_matmul
import torch

test_cases = [
    {
        "name": "small_128",
        "inputs": (
            torch.randn(128, 128, device='cuda', dtype=torch.float32),
            torch.randn(128, 128, 128, device='cuda', dtype=torch.float32),
        ),
        "dtype": torch.float32,
    },
    {
        "name": "medium_256",
        "inputs": (
            torch.randn(256, 256, device='cuda', dtype=torch.float32),
            torch.randn(256, 256, 256, device='cuda', dtype=torch.float32),
        ),
        "dtype": torch.float32,
    },
    {
        "name": "rect_64x32x128",
        "inputs": (
            torch.randn(64, 128, device='cuda', dtype=torch.float32),
            torch.randn(64, 32, 128, device='cuda', dtype=torch.float32),
        ),
        "dtype": torch.float32,
    },
    {
        "name": "large_512",
        "inputs": (
            torch.randn(512, 256, device='cuda', dtype=torch.float32),
            torch.randn(512, 64, 256, device='cuda', dtype=torch.float32),
        ),
        "dtype": torch.float32,
    },
]

run_tilelang_test("batched_matmul", test_cases, pytorch_fn, tilelang_batched_matmul, loose_tol=True)
