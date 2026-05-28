import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from test_utils import run_test, run_tilelang_test
from pytorch_impl import kernel_ff as pytorch_fn
from triton_impl import kernel_ff as triton_fn
from tilelang_impl import kernel_ff as tilelang_kernel_ff
import torch

torch.manual_seed(42)

def make_inputs(batch, seq_len, dim, n_out=None):
    if n_out is None:
        n_out = dim
    x = torch.randn((batch, seq_len, dim), dtype=torch.float16, device='cuda')
    w1 = torch.randn((n_out, dim), dtype=torch.float16, device='cuda')
    w3 = torch.randn((n_out, dim), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((dim,), dtype=torch.float16, device='cuda')
    return (x, w1, w3, rms_w)

test_cases = [
    {"name": "2x4x64", "inputs": make_inputs(2, 4, 64), "dtype": torch.float16},
    {"name": "3x4x64", "inputs": make_inputs(3, 4, 64), "dtype": torch.float16},
    {"name": "2x8x64", "inputs": make_inputs(2, 8, 64), "dtype": torch.float16},
    {"name": "1x16x128", "inputs": make_inputs(1, 16, 128), "dtype": torch.float16},
]

run_tilelang_test("linear_activation", test_cases, pytorch_fn, tilelang_kernel_ff, loose_tol=True)
