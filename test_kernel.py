"""
kernels/gemm/test_kernel.py  —  example test file showing the expected pattern.
Copy this pattern for every new kernel.

Each test_kernel.py must:
  1. Import pytorch_ref, triton_impl, and tilelang_impl from this directory
  2. Define at least 4 test shapes
  3. Compare tilelang output against pytorch reference
  4. Exit 0 on full pass, 1 on any failure
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))           # local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))

import torch
from test_harness import run_test

# ── import implementations ────────────────────────────────────────────────────
import pytorch_ref   as ref_mod
import tilelang_impl as tl_mod

# ── shapes to test ────────────────────────────────────────────────────────────
SHAPES = [
    {"M": 64,   "N": 64,   "K": 64,   "dtype": torch.float16},  # small / sanity
    {"M": 256,  "N": 256,  "K": 256,  "dtype": torch.float16},  # medium
    {"M": 1024, "N": 1024, "K": 1024, "dtype": torch.float16},  # large
    {"M": 512,  "N": 768,  "K": 384,  "dtype": torch.float16},  # non-square
]

# ── input factory ─────────────────────────────────────────────────────────────
def make_inputs(shape):
    M, N, K, dtype = shape["M"], shape["N"], shape["K"], shape["dtype"]
    device = "cuda"
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    return {"A": A, "B": B, "M": M, "N": N, "K": K}

# ── wrappers ──────────────────────────────────────────────────────────────────
def pytorch_fn(inputs):
    return ref_mod.run(inputs["A"], inputs["B"])

def tilelang_fn(inputs):
    return tl_mod.run(inputs["A"], inputs["B"])

# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok = run_test(
        kernel_name="gemm",
        shapes=SHAPES,
        pytorch_fn=pytorch_fn,
        tilelang_fn=tilelang_fn,
        make_inputs=make_inputs,
        loose_tol=False,
    )
    sys.exit(0 if ok else 1)
