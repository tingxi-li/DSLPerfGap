import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import matmul as pytorch_matmul
from triton_impl import matmul as triton_matmul
from test_utils import run_test, run_tilelang_test
from tilelang_impl import matmul as tilelang_matmul


def matmul_compare(ref, test, dtype, loose):
    """Custom comparison for fp16 matmul: use relative tolerance of 1e-2
    since different tiling strategies cause accumulation order differences."""
    ref_f = ref.float()
    test_f = test.float()
    max_err = (ref_f - test_f).abs().max().item()
    # For fp16 matmul, different tiling causes accumulation order differences.
    # Values near 128 have fp16 ULP of 0.125, so we need generous atol.
    passed = torch.allclose(ref_f, test_f, atol=0.2, rtol=1e-2)
    return passed, max_err


def make_case(name, M, N, K, dtype=torch.float16):
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    return {"name": name, "inputs": (a, b), "dtype": dtype}


test_cases = [
    make_case("64x64x64", 64, 64, 64),
    make_case("256x256x256", 256, 256, 256),
    make_case("1024x1024x1024", 1024, 1024, 1024),
    make_case("128x256x512", 128, 256, 512),
    make_case("4096x4096x4096", 4096, 4096, 4096),
]

run_tilelang_test("matmul", test_cases, pytorch_matmul, tilelang_matmul,
                  loose_tol=True, compare_fn=matmul_compare)
