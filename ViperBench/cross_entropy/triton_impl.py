import torch
import triton
import triton.language as tl

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("cross_entropy", "triton") or {}
except Exception:
    _TUNED = {}

@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,
    lse_ptr,
    z_loss_ptr,
    logits_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,
    n_cols,
    n_rows,
    logits_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignored_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(
            n_cols, (col_block_idx + 1) * BLOCK_SIZE
        ):
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (
                    (lse if not SPLIT else 0.0)
                    - smoothing * sum_logits / total_classes
                    - (1 - smoothing) * logits_label
                )
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        else:
            if HAS_SMOOTHING:
                loss = smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
            else:
                loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)


@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,
    n_cols,
    logits_row_stride,
    dlogits_row_stride,
    dloss_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignored_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, (dloss * logit_scale) * probs, mask=col_offsets < n_cols)


def cross_entropy_fwd(
    logits, labels, smoothing, logit_scale, lse_square_scale,
    ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING, SPLIT
):
    n_rows, n_cols = logits.shape
    loss = torch.empty((n_rows, n_cols), dtype=torch.float32, device=logits.device)
    lse = torch.empty((n_rows, n_cols), dtype=torch.float32, device=logits.device)
    z_loss = torch.empty((n_rows, n_cols), dtype=torch.float32, device=logits.device)

    grid = (n_rows, (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)

    cross_entropy_fwd_kernel[grid](
        loss, lse, z_loss, logits, labels, smoothing, logit_scale, lse_square_scale,
        ignored_index, total_classes, class_start_idx, n_cols, n_rows,
        logits.stride(0), BLOCK_SIZE, HAS_SMOOTHING, SPLIT
    )

    return loss, lse, z_loss


def cross_entropy_bwd(
    dloss, logits, lse, labels, smoothing, logit_scale, lse_square_scale,
    ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING
):
    n_rows, n_cols = logits.shape
    dlogits = torch.empty_like(logits)

    grid = (n_rows, (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)

    cross_entropy_bwd_kernel[grid](
        dlogits, dloss, logits, lse, labels, smoothing, logit_scale, lse_square_scale,
        ignored_index, total_classes, class_start_idx, n_cols,
        logits.stride(0), dlogits.stride(0), dloss.stride(0), BLOCK_SIZE, HAS_SMOOTHING
    )

    return dlogits
