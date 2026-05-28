import torch
import triton
import triton.language as tl

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("mean_reduction", "triton") or {}
except Exception:
    _TUNED = {}


@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean, mean, row_mask)


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def mean_reduction(input_tensor, dim, keepdim=False, dtype=None):
    x = input_tensor
    if dtype is None:
        dtype = x.dtype

    shape = list(x.shape)
    if isinstance(dim, int):
        dim = [dim]
    dim = [d % x.ndim for d in dim]
    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    out = torch.empty(shape, dtype=dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    with torch.cuda.device(x.device):
        mean_dim_kernel[grid](x, out, M, N, BLOCK_M=_TUNED.get("BLOCK_M", 8), BLOCK_N=_TUNED.get("BLOCK_N", 8))
    if not keepdim:
        out = out.squeeze(dim)
    return out
