import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import layer_norm as pytorch_layer_norm
from triton_impl import layer_norm as triton_layer_norm
from test_utils import run_test, run_tilelang_test
from tilelang_impl import layer_norm as tilelang_layer_norm


def make_case(name, shape, D, dtype=torch.bfloat16):
    x = torch.randn(shape, device='cuda', dtype=dtype)
    weight = torch.randn(D, device='cuda', dtype=dtype)
    bias = torch.randn(D, device='cuda', dtype=dtype)
    return {"name": name, "inputs": (x, weight, bias), "dtype": dtype}


test_cases = [
    make_case("128x1024_bf16", (128, 1024), 1024, torch.bfloat16),
    make_case("256x2048_bf16", (256, 2048), 2048, torch.bfloat16),
    make_case("64x4096_bf16", (64, 4096), 4096, torch.bfloat16),
    make_case("512x1024_bf16", (512, 1024), 1024, torch.bfloat16),
    make_case("3d_4x32x1024_bf16", (4, 32, 1024), 1024, torch.bfloat16),
    make_case("3d_2x8x2048_bf16", (2, 8, 2048), 2048, torch.bfloat16),
]

run_tilelang_test("layer_norm", test_cases, pytorch_layer_norm, tilelang_layer_norm, loose_tol=True)
