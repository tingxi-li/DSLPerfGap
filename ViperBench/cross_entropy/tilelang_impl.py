"""
TileLang implementation of cross-entropy forward and backward kernels.
Uses @tilelang.jit decorator with T.prim_func.
"""
import torch
import tilelang
import tilelang.language as T

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("cross_entropy", "tilelang") or {}
except Exception:
    _TUNED = {}

_threads = _TUNED.get("threads", 32)


def _make_fwd_kernel():
    @tilelang.jit
    def fwd_kernel(
        _n_rows, _n_cols, _BLOCK_SIZE, _num_col_blocks, _out_size,
        _HAS_SMOOTHING, _SPLIT, _ignored_index, _total_classes, _class_start_idx,
    ):
        @T.prim_func
        def kernel(
            logits: T.Tensor((_n_rows, _n_cols), "float32"),
            labels: T.Tensor((_n_rows,), "int32"),
            params: T.Tensor((3,), "float32"),
            loss_out: T.Tensor((_out_size,), "float32"),
            lse_out: T.Tensor((_out_size,), "float32"),
            z_loss_out: T.Tensor((_out_size,), "float32"),
        ):
            with T.Kernel(_n_rows, _num_col_blocks, threads=_threads) as (row_idx, col_block_idx):
                local_max = T.alloc_local((1,), "float32")
                local_sum = T.alloc_local((1,), "float32")
                local_sum_logits = T.alloc_local((1,), "float32")

                local_max[0] = T.float32(-1e30)
                local_sum[0] = T.float32(0.0)
                local_sum_logits[0] = T.float32(0.0)

                col_start = col_block_idx * _BLOCK_SIZE

                smoothing_f = params[0]
                logit_scale_f = params[1]
                lse_sq_scale_f = params[2]

                # Pass 1: find max over block
                for v in T.serial(_BLOCK_SIZE):
                    col = col_start + v
                    val = T.if_then_else(
                        col < _n_cols,
                        logits[row_idx, col] * logit_scale_f,
                        T.float32(-1e30)
                    )
                    local_max[0] = T.max(local_max[0], val)

                # Pass 2: sum exp and sum logits
                for v in T.serial(_BLOCK_SIZE):
                    col = col_start + v
                    val = T.if_then_else(
                        col < _n_cols,
                        logits[row_idx, col] * logit_scale_f,
                        T.float32(-1e30)
                    )
                    local_sum[0] = local_sum[0] + T.exp(val - local_max[0])
                    local_sum_logits[0] = local_sum_logits[0] + T.if_then_else(
                        col < _n_cols, val, T.float32(0.0)
                    )

                lse_val = T.log(local_sum[0]) + local_max[0]
                flat_idx = col_block_idx * _n_rows + row_idx
                lse_out[flat_idx] = lse_val

                label_raw = labels[row_idx]
                adj_label = label_raw - _class_start_idx

                label_in_block = (adj_label >= col_start) & (
                    adj_label < T.min(_n_cols, col_start + _BLOCK_SIZE)
                )

                safe_col = T.if_then_else(label_in_block, adj_label, T.int32(0))
                logits_label = logits[row_idx, safe_col] * logit_scale_f

                lse_or_zero = T.if_then_else(_SPLIT == 0, lse_val, T.float32(0.0))
                total_classes_f = T.Cast("float32", _total_classes)

                # loss when label is in this block, with smoothing
                loss_in_smooth = (
                    lse_or_zero
                    - smoothing_f * local_sum_logits[0] / total_classes_f
                    - (T.float32(1.0) - smoothing_f) * logits_label
                )
                loss_in_nosmooth = lse_or_zero - logits_label
                loss_in = T.if_then_else(
                    _HAS_SMOOTHING == 1, loss_in_smooth, loss_in_nosmooth
                )

                # loss when label is NOT in this block
                loss_out_smooth = smoothing_f * (
                    lse_or_zero - local_sum_logits[0] / total_classes_f
                )
                loss_out_val = T.if_then_else(
                    _HAS_SMOOTHING == 1, loss_out_smooth, T.float32(0.0)
                )

                loss_not_ignored = T.if_then_else(label_in_block, loss_in, loss_out_val)

                z_loss_val = T.if_then_else(
                    _SPLIT == 0,
                    lse_sq_scale_f * lse_val * lse_val,
                    T.float32(0.0),
                )

                loss_with_z = T.if_then_else(
                    _SPLIT == 0, loss_not_ignored + z_loss_val, loss_not_ignored
                )

                final_loss = T.if_then_else(
                    label_raw == _ignored_index, T.float32(0.0), loss_with_z
                )
                final_z = T.if_then_else(
                    label_raw == _ignored_index, T.float32(0.0), z_loss_val
                )

                loss_out[flat_idx] = final_loss
                z_loss_out[flat_idx] = final_z

        return kernel

    return fwd_kernel


def _make_bwd_kernel():
    @tilelang.jit
    def bwd_kernel(
        _n_rows, _n_cols, _BLOCK_SIZE, _num_col_blocks,
        _HAS_SMOOTHING, _ignored_index, _total_classes, _class_start_idx,
    ):
        @T.prim_func
        def kernel(
            dloss: T.Tensor((_n_rows,), "float32"),
            logits: T.Tensor((_n_rows, _n_cols), "float32"),
            lse: T.Tensor((_n_rows,), "float32"),
            labels: T.Tensor((_n_rows,), "int32"),
            params: T.Tensor((3,), "float32"),
            dlogits: T.Tensor((_n_rows, _n_cols), "float32"),
        ):
            with T.Kernel(_n_rows, _num_col_blocks, threads=_threads) as (row_idx, col_block_idx):
                col_start = col_block_idx * _BLOCK_SIZE

                smoothing_f = params[0]
                logit_scale_f = params[1]
                lse_sq_scale_f = params[2]

                label_raw = labels[row_idx]
                dl = T.if_then_else(
                    label_raw != _ignored_index, dloss[row_idx], T.float32(0.0)
                )
                lse_val = lse[row_idx]
                adj_label = label_raw - _class_start_idx
                smooth_neg = smoothing_f / T.Cast("float32", _total_classes)

                for v in T.serial(_BLOCK_SIZE):
                    col = col_start + v
                    logit_val = T.if_then_else(
                        col < _n_cols,
                        logits[row_idx, col] * logit_scale_f,
                        T.float32(-1e30),
                    )
                    prob = T.exp(logit_val - lse_val)
                    prob = prob + T.float32(2.0) * lse_sq_scale_f * lse_val * prob

                    # Subtract 1 at label position (or 1-smoothing if smoothing)
                    prob_adj = T.if_then_else(
                        _HAS_SMOOTHING == 1,
                        T.if_then_else(
                            col == adj_label,
                            prob - (T.float32(1.0) - smoothing_f),
                            prob,
                        )
                        - smooth_neg,
                        T.if_then_else(col == adj_label, prob - T.float32(1.0), prob),
                    )

                    result = (dl * logit_scale_f) * prob_adj

                    # Only write valid columns using if_then_else guard
                    # We write to dlogits[row_idx, col] only if col < n_cols
                    # Since we can't conditionally skip a store, we write to a
                    # safe location (col 0) when out of bounds, but the valid
                    # write will overwrite it.
                    # Actually, let's just always write - out of bounds won't happen
                    # if BLOCK_SIZE is reasonable and we guard properly.
                    safe_write_col = T.if_then_else(col < _n_cols, col, T.int32(0))
                    # We do a conditional write: only meaningful if col < n_cols
                    dlogits[row_idx, safe_write_col] = T.if_then_else(
                        col < _n_cols, result, dlogits[row_idx, safe_write_col]
                    )

        return kernel

    return bwd_kernel


# Module-level cached kernel factories
_fwd_kernel = _make_fwd_kernel()
_bwd_kernel = _make_bwd_kernel()


def cross_entropy_fwd(
    logits,
    labels,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,
    BLOCK_SIZE,
    HAS_SMOOTHING,
    SPLIT,
):
    n_rows, n_cols = logits.shape
    num_col_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    out_size = n_rows * n_cols

    logits_f32 = logits.float().contiguous()
    labels_i32 = labels.int().contiguous()

    params = torch.tensor(
        [smoothing, logit_scale, lse_square_scale],
        dtype=torch.float32,
        device=logits.device,
    )

    loss_out = torch.zeros(out_size, dtype=torch.float32, device=logits.device)
    lse_out = torch.zeros(out_size, dtype=torch.float32, device=logits.device)
    z_loss_out = torch.zeros(out_size, dtype=torch.float32, device=logits.device)

    _fwd_kernel(
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_col_blocks,
        out_size,
        int(HAS_SMOOTHING),
        int(SPLIT),
        ignored_index,
        total_classes,
        class_start_idx,
    )(logits_f32, labels_i32, params, loss_out, lse_out, z_loss_out)

    loss_2d = loss_out.reshape(n_rows, n_cols)
    lse_2d = lse_out.reshape(n_rows, n_cols)
    z_loss_2d = z_loss_out.reshape(n_rows, n_cols)

    return loss_2d, lse_2d, z_loss_2d


def cross_entropy_bwd(
    dloss,
    logits,
    lse,
    labels,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,
    BLOCK_SIZE,
    HAS_SMOOTHING,
):
    n_rows, n_cols = logits.shape
    num_col_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    logits_f32 = logits.float().contiguous()
    labels_i32 = labels.int().contiguous()
    dloss_f32 = dloss.float().contiguous()
    lse_f32 = lse.float().contiguous()

    params = torch.tensor(
        [smoothing, logit_scale, lse_square_scale],
        dtype=torch.float32,
        device=logits.device,
    )

    dlogits = torch.zeros_like(logits_f32)

    _bwd_kernel(
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_col_blocks,
        int(HAS_SMOOTHING),
        ignored_index,
        total_classes,
        class_start_idx,
    )(dloss_f32, logits_f32, lse_f32, labels_i32, params, dlogits)

    return dlogits.to(logits.dtype)
