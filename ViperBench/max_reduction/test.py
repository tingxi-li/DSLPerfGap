import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import max_reduction as pytorch_max_reduction
from triton_impl import max_reduction as triton_max_reduction
from tilelang_impl import max_reduction as tilelang_max_reduction
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
        "name": "2D_dim1_keepdim",
        "inputs": (torch.randn(64, 256, device='cuda', dtype=torch.float32), 1, True),
        "dtype": torch.float32,
    },
    {
        "name": "3D_neg_dim",
        "inputs": (torch.randn(16, 32, 64, device='cuda', dtype=torch.float32), -1),
        "dtype": torch.float32,
    },
]

def compare_max(ref, test, dtype, loose):
    """Compare (values, indices) tuples - values must match, indices must match."""
    from test_utils import compare_tensors
    ref_vals, ref_idx = ref
    test_vals, test_idx = test
    val_pass, val_err = compare_tensors(ref_vals, test_vals, dtype, loose)
    idx_pass = torch.equal(ref_idx, test_idx)
    idx_err = (ref_idx.float() - test_idx.float()).abs().max().item() if not idx_pass else 0.0
    return val_pass and idx_pass, max(val_err, idx_err)

run_tilelang_test("max_reduction", test_cases, pytorch_max_reduction, tilelang_max_reduction, compare_fn=compare_max)
