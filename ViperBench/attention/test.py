import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from test_utils import run_test, run_tilelang_test
from pytorch_impl import attention_fwd as pytorch_fn
from triton_impl import attention_fwd as triton_fn
from tilelang_impl import attention_fwd as tilelang_fn
import torch

torch.manual_seed(0)

def make_qkv(B, H, T, D, dtype=torch.float32):
    q = torch.randn(B, H, T, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, T, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, T, D, device='cuda', dtype=dtype)
    return (q, k, v)

# Use smaller sizes first to ensure correctness
q1, k1, v1 = make_qkv(1, 2, 64, 32)
q2, k2, v2 = make_qkv(2, 4, 128, 64)
q3, k3, v3 = make_qkv(1, 2, 64, 32)
q4, k4, v4 = make_qkv(1, 2, 32, 32)

q5, k5, v5 = make_qkv(1, 2, 64, 32)

test_cases = [
    {"name": "basic_no_flags", "inputs": (q1, k1, v1), "dtype": torch.float32},
    {"name": "medium_no_flags", "inputs": (q2, k2, v2), "dtype": torch.float32},
    {"name": "ifcond_true", "inputs": (q3, k3, v3, False, True), "dtype": torch.float32},
    {"name": "small_seq", "inputs": (q4, k4, v4, False, True), "dtype": torch.float32},
    {"name": "store_true", "inputs": (q5, k5, v5, True, False), "dtype": torch.float32},
]

run_tilelang_test("attention", test_cases, pytorch_fn, tilelang_fn, loose_tol=True)
