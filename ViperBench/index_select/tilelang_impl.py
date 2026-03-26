import tilelang
import tilelang.language as T
import torch

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("index_select", "tilelang") or {}
except Exception:
    _TUNED = {}


@tilelang.jit
def _index_select_kernel_f32(n, d, s, bN=_TUNED.get("block_N", 32)):
    @T.prim_func
    def func(
        Src: T.Tensor((s, d), "float32"),
        Idx: T.Tensor((n,), "int32"),
        Out: T.Tensor((n, d), "float32"),
    ):
        with T.Kernel(T.ceildiv(n, bN), threads=_TUNED.get("threads", 128)) as bx:
            idx_frag = T.alloc_fragment((bN,), "int32")
            out_frag = T.alloc_fragment((bN, d), "float32")
            T.copy(Idx[bx * bN], idx_frag)
            for i, j in T.Parallel(bN, d):
                out_frag[i, j] = Src[idx_frag[i], j]
            T.copy(out_frag, Out[bx * bN, 0])
    return func


@tilelang.jit
def _index_select_kernel_f16(n, d, s, bN=_TUNED.get("block_N", 32)):
    @T.prim_func
    def func(
        Src: T.Tensor((s, d), "float16"),
        Idx: T.Tensor((n,), "int32"),
        Out: T.Tensor((n, d), "float16"),
    ):
        with T.Kernel(T.ceildiv(n, bN), threads=_TUNED.get("threads", 128)) as bx:
            idx_frag = T.alloc_fragment((bN,), "int32")
            out_frag = T.alloc_fragment((bN, d), "float16")
            T.copy(Idx[bx * bN], idx_frag)
            for i, j in T.Parallel(bN, d):
                out_frag[i, j] = Src[idx_frag[i], j]
            T.copy(out_frag, Out[bx * bN, 0])
    return func


def index_select(output, source, index):
    """
    Index-select rows from source by index using TileLang.
    output[:] = source[index] — row gather on dim 0.
    Supports ND source tensors: flattens to 2D (keeping dim 0), runs kernel, reshapes back.
    Index must be 1D.
    """
    if not index.ndim == 1:
        raise ValueError(f"index must be 1D, got {index.ndim}D tensor with shape {index.shape}")

    orig_shape = source.shape
    orig_dtype = source.dtype

    # Flatten to 2D keeping dim 0
    if source.ndim > 2:
        source_2d = source.reshape(orig_shape[0], -1).contiguous()
    elif source.ndim == 1:
        source_2d = source.unsqueeze(1).contiguous()
    else:
        source_2d = source.contiguous()

    N = index.shape[0]          # number of indices
    D = source_2d.shape[1]      # number of columns
    S = source_2d.shape[0]      # number of source rows

    # Choose kernel and working dtype based on input dtype
    if orig_dtype == torch.float16:
        working_dtype = torch.float16
        kernel_fn = _index_select_kernel_f16
        src_work = source_2d
    elif orig_dtype == torch.float32:
        working_dtype = torch.float32
        kernel_fn = _index_select_kernel_f32
        src_work = source_2d
    else:
        # Fallback: convert to float32 for unsupported dtypes (bf16, etc.)
        working_dtype = torch.float32
        kernel_fn = _index_select_kernel_f32
        src_work = source_2d.float()

    idx_i32 = index.to(torch.int32).contiguous()

    # Pad N to multiple of block_N (minimum 32 to avoid degenerate cases)
    block_N = _TUNED.get("block_N", 32)
    N_pad = ((N + block_N - 1) // block_N) * block_N

    # Pad D to multiple of 128 to avoid layout inference issues
    D_align = 128
    D_pad = ((D + D_align - 1) // D_align) * D_align

    # Pad source columns if needed
    if D_pad != D:
        src_padded = torch.zeros(S, D_pad, device=source.device, dtype=working_dtype)
        src_padded[:, :D] = src_work
    else:
        src_padded = src_work

    # Pad indices if needed
    if N_pad != N:
        idx_pad = torch.zeros(N_pad, device=index.device, dtype=torch.int32)
        idx_pad[:N] = idx_i32
    else:
        idx_pad = idx_i32

    out_pad = torch.zeros(N_pad, D_pad, device=source.device, dtype=working_dtype)
    func = kernel_fn(N_pad, D_pad, S, bN=block_N)
    func(src_padded, idx_pad, out_pad)

    # Slice off padding, reshape, cast back
    result = out_pad[:N, :D]
    if source.ndim > 2:
        out_shape = (N,) + orig_shape[1:]
        result = result.reshape(out_shape)
    elif source.ndim == 1:
        result = result.squeeze(1)

    if result.dtype != orig_dtype:
        result = result.to(orig_dtype)
    output.copy_(result)
    return output
