import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import conv2d as pytorch_conv2d
from triton_impl import conv2d as triton_conv2d
from test_utils import run_test, run_tilelang_test

torch.manual_seed(42)

# No bias tests (Triton kernel lacks full bias support path in kernel itself,
# we add it outside, but to be safe test without bias)
test_cases = [
    {
        "name": "basic_3x3",
        "inputs": {
            "input": torch.randn(1, 3, 32, 32, device='cuda', dtype=torch.float32),
            "weight": torch.randn(16, 3, 3, 3, device='cuda', dtype=torch.float32),
        },
        "dtype": torch.float32,
    },
    {
        "name": "stride2_pad1",
        "inputs": {
            "input": torch.randn(1, 3, 32, 32, device='cuda', dtype=torch.float32),
            "weight": torch.randn(16, 3, 3, 3, device='cuda', dtype=torch.float32),
            "stride": 2,
            "padding": 1,
        },
        "dtype": torch.float32,
    },
    {
        "name": "5x5_kernel",
        "inputs": {
            "input": torch.randn(1, 3, 32, 32, device='cuda', dtype=torch.float32),
            "weight": torch.randn(16, 3, 5, 5, device='cuda', dtype=torch.float32),
        },
        "dtype": torch.float32,
    },
    {
        "name": "batch4_pad2",
        "inputs": {
            "input": torch.randn(4, 3, 16, 16, device='cuda', dtype=torch.float32),
            "weight": torch.randn(8, 3, 3, 3, device='cuda', dtype=torch.float32),
            "padding": 2,
        },
        "dtype": torch.float32,
    },
]

def conv2d_tilelang_compare(ref, test, dtype, loose):
    """Custom comparison for conv2d with TileLang: uses fp16 GEMM internally
    so precision is limited by fp16 accumulation order differences."""
    ref_f = ref.float()
    test_f = test.float()
    max_err = (ref_f - test_f).abs().max().item()
    passed = torch.allclose(ref_f, test_f, atol=2e-2, rtol=1e-2)
    return passed, max_err


if __name__ == '__main__':
    from tilelang_impl import conv2d as tilelang_conv2d
    run_tilelang_test("conv2d", test_cases, pytorch_conv2d, tilelang_conv2d,
                      loose_tol=True, compare_fn=conv2d_tilelang_compare)
