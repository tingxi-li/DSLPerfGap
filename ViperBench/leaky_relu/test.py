import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import leaky_relu as pytorch_leaky_relu
from tilelang_impl import leaky_relu as tilelang_leaky_relu
from test_utils import run_tilelang_test

torch.manual_seed(42)

M1, K1, N1 = 64, 128, 64
a1 = torch.randn((M1, K1), device='cuda', dtype=torch.float16)
b1 = torch.randn((K1, N1), device='cuda', dtype=torch.float16)

M2, K2, N2 = 128, 256, 128
a2 = torch.randn((M2, K2), device='cuda', dtype=torch.float16)
b2 = torch.randn((K2, N2), device='cuda', dtype=torch.float16)

M3, K3, N3 = 32, 32, 32
a3 = torch.randn((M3, K3), device='cuda', dtype=torch.float16)
b3 = torch.randn((K3, N3), device='cuda', dtype=torch.float16)

M4, K4, N4 = 256, 512, 256
a4 = torch.randn((M4, K4), device='cuda', dtype=torch.float16)
b4 = torch.randn((K4, N4), device='cuda', dtype=torch.float16)

test_cases = [
    {"name": "small_leaky_relu", "inputs": (a1, b1, "leaky_relu"), "dtype": torch.float16},
    {"name": "medium_leaky_relu", "inputs": (a2, b2, "leaky_relu"), "dtype": torch.float16},
    {"name": "small_no_activation", "inputs": (a3, b3, ""), "dtype": torch.float16},
    {"name": "large_leaky_relu", "inputs": (a4, b4, "leaky_relu"), "dtype": torch.float16},
]

run_tilelang_test("leaky_relu", test_cases, pytorch_leaky_relu, tilelang_leaky_relu)
