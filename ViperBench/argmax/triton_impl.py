import torch
import triton
import triton.language as tl
import math

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("argmax", "triton") or {}
except Exception:
    _TUNED = {}


def can_use_int32_index(tensor):
    return tensor.numel() < 2**31


@triton.jit
def argmax_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if INT64_INDEX:
        pid_m = pid_m.to(tl.int64)
        pid_k = pid_k.to(tl.int64)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    max_values = tl.full([BLOCK_M], dtype=tl.float32, value=float("-inf"))
    argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        local_max, local_argmax = tl.max(
            inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, start_n + local_argmax, argmax_values)

    offset_index = m_offset * K + pid_k
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_index_ptrs, argmax_values, mask=mask1)


def argmax(input_tensor, dim, keepdim=False):
    inp = input_tensor
    assert dim is not None, "dim must be specified"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()
    use_int64_index = not can_use_int32_index(inp)

    shape_list = list(shape)
    shape_list[dim] = 1
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
    if not keepdim:
        out_index = torch.squeeze(out_index, dim)

    BLOCK_M = _TUNED.get("BLOCK_M", 128)
    BLOCK_N = _TUNED.get("BLOCK_N", 128)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch.cuda.device(inp.device):
        argmax_kernel[grid](
            inp,
            out_index,
            M,
            N,
            K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            INT64_INDEX=use_int64_index,
        )

    return out_index
