import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import mean_reduction as pytorch_mean_reduction
from tilelang_impl import mean_reduction as tilelang_mean_reduction
from test_utils import run_tilelang_test

torch.manual_seed(42)

test_cases = [
    {
        "name": "2D_dim0",
        "inputs": (torch.randn(64, 128, device='cuda', dtype=torch.float32), 0),
        "dtype": torch.float32,
    },
    {
        "name": "2D_dim1_keepdim",
        "inputs": (torch.randn(64, 128, device='cuda', dtype=torch.float32), 1, True),
        "dtype": torch.float32,
    },
    {
        "name": "4D_multi_dim",
        "inputs": (torch.randn(2, 3, 4, 5, device='cuda', dtype=torch.float32), [1, 2]),
        "dtype": torch.float32,
    },
    {
        "name": "3D_dim2",
        "inputs": (torch.randn(8, 16, 32, device='cuda', dtype=torch.float32), 2),
        "dtype": torch.float32,
    },
]

run_tilelang_test("mean_reduction", test_cases, pytorch_mean_reduction, tilelang_mean_reduction, loose_tol=True)
