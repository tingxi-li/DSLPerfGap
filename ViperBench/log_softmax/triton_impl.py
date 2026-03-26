import logging

import torch
import triton
import triton.language as tl

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("log_softmax", "triton") or {}
except Exception:
    _TUNED = {}


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


def heur_num_warps(args):
    if args["N"] <= 1024:
        return 4
    elif args["N"] <= 2048:
        return 8
    else:
        return 16


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}),
        triton.Config({"BLOCK_M": 2}),
        triton.Config({"BLOCK_M": 4}),
        triton.Config({"BLOCK_M": 8}),
    ],
    key=["M", "N"],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "num_warps": heur_num_warps,
    }
)
@triton.jit
def log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = tl.log(numerator / denominator)
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, softmax_output, mask=mask)


class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim = dim % x.ndim
        M = 1
        N = x.shape[dim]
        for i in range(dim):
            M *= x.shape[i]
        inp = x.contiguous()
        if dtype is None:
            dtype = x.dtype
        out = torch.empty_like(inp, dtype=dtype)
        K = inp.numel() // M // N

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        with torch.cuda.device(inp.device):
            log_softmax_kernel[grid](out, inp, M, N, K)
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        return out_grad, None, None


def log_softmax(x, dim=-1, dtype=None):
    return LogSoftmax.apply(x, dim, dtype)
