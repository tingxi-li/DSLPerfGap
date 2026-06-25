import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("max_reduction", "triton") or {}
except Exception:
    _TUNED = {}


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Tiled row reduction: each program owns BLOCK_M rows (for a fixed k) and
    # streams the reduced dim in BLOCK_N-wide tiles, keeping a running max and
    # argmax. The old kernel loaded the whole N=next_pow2(N) row in one block,
    # which spilled badly for large N. strict-> tie-break matches torch.max.
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    acc_val = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    acc_idx = tl.zeros((BLOCK_M,), tl.int64)
    for n0 in range(0, N, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        cmask = cols < N
        ptrs = inp + rows[:, None] * N * K + cols[None, :] * K + pid_k
        vals = tl.load(ptrs, mask=rmask[:, None] & cmask[None, :], other=-float("inf")).to(tl.float32)
        tmax, targ = tl.max(vals, axis=1, return_indices=True)
        tidx = n0 + targ
        upd = tmax > acc_val
        acc_idx = tl.where(upd, tidx, acc_idx)
        acc_val = tl.where(upd, tmax, acc_val)

    oidx = rows * K + pid_k
    tl.store(out_value + oidx, acc_val.to(out_value.dtype.element_ty), mask=rmask)
    tl.store(out_index + oidx, acc_idx, mask=rmask)


def max_reduction(input, dim, keepdim=False):
    inp = input
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch.cuda.device(inp.device):
        max_kernel[grid](inp, out_value, out_index, M, N, K)
    return (out_value, out_index)
