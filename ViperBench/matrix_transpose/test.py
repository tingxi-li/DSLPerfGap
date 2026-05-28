import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import matrix_transpose as pytorch_transpose
from triton_impl import matrix_transpose as triton_transpose
from tilelang_impl import matrix_transpose as tilelang_matrix_transpose
from test_utils import run_test, run_tilelang_test


def make_case(name, M, N, dtype=torch.float32):
    x = torch.randn((M, N), device='cuda', dtype=dtype)
    return {"name": name, "inputs": (x,), "dtype": dtype}


test_cases = [
    make_case("32x32", 32, 32),
    make_case("128x256", 128, 256),
    make_case("1024x1024", 1024, 1024),
    make_case("64x512", 64, 512),
    make_case("1000x700", 1000, 700),
    make_case("37x73_odd", 37, 73),
    make_case("1x1000_single_row", 1, 1000),
    make_case("1000x1_single_col", 1000, 1),
    make_case("7x13_small_odd", 7, 13),
]

run_tilelang_test("matrix_transpose", test_cases, pytorch_transpose, tilelang_matrix_transpose)
