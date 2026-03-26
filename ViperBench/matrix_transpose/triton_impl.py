import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    M_ptr,
    Out_ptr,
    matrix_stridex,
    matrix_stridey,
    out_stridex,
    out_stridey,
    n_rows,
    n_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_offs = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offs = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    row_mask = row_offs < n_rows
    col_mask = col_offs < n_cols

    # Load from input[row, col]
    in_ptrs = M_ptr + row_offs[:, None] * matrix_stridex + col_offs[None, :] * matrix_stridey
    mask = row_mask[:, None] & col_mask[None, :]
    vals = tl.load(in_ptrs, mask=mask)

    # Store to output[col, row] — transpose
    out_ptrs = Out_ptr + col_offs[None, :] * out_stridex + row_offs[:, None] * out_stridey
    tl.store(out_ptrs, vals, mask=mask)


def matrix_transpose(x):
    """Unified API: matrix_transpose(x) -> Tensor, returns x.T contiguous."""
    if x.ndim != 2:
        raise ValueError(f"matrix_transpose requires a 2D tensor, got {x.ndim}D tensor with shape {x.shape}")
    M, N = x.shape
    out = torch.empty((N, M), device=x.device, dtype=x.dtype)

    BLOCK_ROWS = 32
    BLOCK_COLS = 32

    grid = (triton.cdiv(M, BLOCK_ROWS), triton.cdiv(N, BLOCK_COLS))
    transpose_kernel[grid](
        x, out,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        M, N,
        BLOCK_ROWS, BLOCK_COLS,
    )
    return out
