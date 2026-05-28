import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import argmax as pytorch_argmax
from triton_impl import argmax as triton_argmax
from tilelang_impl import argmax as tilelang_argmax
from test_utils import run_test, run_tilelang_test

torch.manual_seed(42)

test_cases = [
    {
        "name": "2D_dim0",
        "inputs": (torch.randn(128, 256, device='cuda', dtype=torch.float32), 0),
        "dtype": torch.float32,
    },
    {
        "name": "2D_dim1",
        "inputs": (torch.randn(256, 128, device='cuda', dtype=torch.float32), 1),
        "dtype": torch.float32,
    },
    {
        "name": "3D_dim2",
        "inputs": (torch.randn(32, 64, 128, device='cuda', dtype=torch.float32), 2),
        "dtype": torch.float32,
    },
    {
        "name": "3D_dim1_keepdim",
        "inputs": (torch.randn(16, 32, 64, device='cuda', dtype=torch.float32), 1, True),
        "dtype": torch.float32,
    },
]

def compare_indices(ref, test, dtype, loose):
    """Compare int64 index tensors exactly."""
    passed = torch.equal(ref, test)
    max_err = (ref.float() - test.float()).abs().max().item() if not passed else 0.0
    return passed, max_err

run_tilelang_test("argmax", test_cases, pytorch_argmax, tilelang_argmax, compare_fn=compare_indices)
