import torch
import triton
import triton.language as tl

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("index_select", "triton") or {}
except Exception:
    _TUNED = {}

@triton.jit
def index_select_cat_fwd_kernel(
    output_ptr,
    source_ptr,
    index_ptr,
    num_indices,
    num_cols,
    stride0,
    stride1,
    BLOCK_SIZE_INDEX: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    rows = tl.load(index_ptr + indices, mask=(indices < num_indices))
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    source_offsets = source_ptr + rows[:, None] * stride0 + cols[None, :] * stride1
    mask = (indices[:, None] < num_indices) & (cols[None, :] < num_cols)
    output = tl.load(source_offsets, mask=mask)

    output_offsets = output_ptr + indices[:, None] * stride0 + cols[None, :] * stride1
    tl.store(output_offsets, output, mask=mask)


def index_select(
    output: torch.Tensor,
    source: torch.Tensor,
    index: torch.Tensor,
):
    """
    Index-select rows from source by index using Triton.
    Supports ND source tensors: flattens to 2D (keeping dim 0), runs kernel, reshapes back.
    Index must be 1D.
    """
    if not (source.is_cuda and index.is_cuda):
        raise ValueError("The index tensor and the source tensor must be of type CUDA!")
    if not index.ndim == 1:
        raise ValueError(f"index must be 1D, got {index.ndim}D tensor with shape {index.shape}")

    orig_shape = source.shape
    # Flatten to 2D keeping dim 0
    if source.ndim > 2:
        source_2d = source.reshape(orig_shape[0], -1).contiguous()
    elif source.ndim == 1:
        source_2d = source.unsqueeze(1).contiguous()
    else:
        source_2d = source.contiguous()

    num_rows, num_cols = source_2d.shape
    num_indices = index.shape[0]

    if num_indices > num_rows:
        print(f"Warning: The number of indices exceeds the number of rows in the source tensor. Truncating indices.")
        num_indices = num_rows
        index = index[:num_rows]

    # Allocate 2D output
    output_2d = torch.empty((num_indices, num_cols), device=source.device, dtype=source.dtype)

    stride0, stride1 = source_2d.stride(0), source_2d.stride(1)

    def grid(meta):
        return (
            triton.cdiv(num_indices, meta["BLOCK_SIZE_INDEX"]),
            triton.cdiv(num_cols, meta["BLOCK_SIZE_COL"]),
        )

    index_select_cat_fwd_kernel[grid](
        output_2d,
        source_2d,
        index,
        num_indices,
        num_cols,
        stride0,
        stride1,
        BLOCK_SIZE_INDEX=1,
        BLOCK_SIZE_COL=_TUNED.get("BLOCK_SIZE_COL", 512),
    )

    # Reshape back to ND
    if source.ndim > 2:
        out_shape = (num_indices,) + orig_shape[1:]
        result = output_2d.reshape(out_shape)
    elif source.ndim == 1:
        result = output_2d.squeeze(1)
    else:
        result = output_2d

    output.copy_(result)
    return output
