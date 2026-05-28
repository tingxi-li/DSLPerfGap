import torch


def cross_entropy_fwd(
    logits, labels, smoothing, logit_scale, lse_square_scale,
    ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING, SPLIT
):
    """
    PyTorch reference matching the Triton cross_entropy_fwd_kernel exactly.

    The Triton kernel processes per (row, col_block) and stores results in a flat
    buffer indexed as [col_block_idx * n_rows + row_idx]. This reference reproduces
    that computation and memory layout **vectorized** (whole-tensor ops, no Python
    per-element loop / no `.item()` host syncs), so it is also a *fair* eager-PyTorch
    latency baseline. The earlier version was a triple Python loop whose host syncs
    made it ~10^4x slower than the kernel and produced a meaningless "library
    efficiency"; this version is numerically identical (verified by cross_entropy/
    test.py against both the Triton and TileLang kernels) but fast.

    Returns (loss, lse, z_loss), each shaped [n_rows, n_cols] in the kernel's layout.
    """
    n_rows, n_cols = logits.shape
    dev = logits.device
    num_col_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded = num_col_blocks * BLOCK_SIZE

    # Block view (n_rows, num_col_blocks, BLOCK_SIZE); OOB columns -> -inf, matching
    # the kernel's `tl.load(..., other=-inf)`.
    Lpad = torch.full((n_rows, padded), float("-inf"), dtype=torch.float32, device=dev)
    Lpad[:, :n_cols] = logits.float() * logit_scale
    Lb = Lpad.view(n_rows, num_col_blocks, BLOCK_SIZE)

    # Per-block LSE (each block reduces over only its own BLOCK_SIZE columns).
    max_logits = Lb.max(dim=2).values                                   # (R, B)
    lse = torch.log(torch.exp(Lb - max_logits.unsqueeze(2)).sum(dim=2)) + max_logits

    labels = labels.to(torch.int64)
    adj_label = labels - class_start_idx                                # (R,)
    not_ignored = labels != ignored_index                               # (R,)

    # Which block holds the (shifted) label, with the kernel's exact bounds test:
    #   adj_label >= cb*BS  and  adj_label < min(n_cols, (cb+1)*BS)
    block_ids = torch.arange(num_col_blocks, device=dev)
    col_start = block_ids * BLOCK_SIZE                                   # (B,)
    col_end = torch.clamp((block_ids + 1) * BLOCK_SIZE, max=n_cols)      # (B,)
    in_block = (adj_label.unsqueeze(1) >= col_start) & (adj_label.unsqueeze(1) < col_end)  # (R,B)

    # Label logit (safe-gathered; only consumed where in_block is True).
    safe_label = adj_label.clamp(0, n_cols - 1)
    logits_label = logits.gather(1, safe_label.unsqueeze(1)).float().squeeze(1) * logit_scale  # (R,)

    lse_term = lse if not SPLIT else torch.zeros_like(lse)               # (lse if not SPLIT else 0)

    if HAS_SMOOTHING:
        valid = (torch.arange(padded, device=dev) < n_cols).view(1, num_col_blocks, BLOCK_SIZE)
        sum_logits = torch.where(valid, Lb, torch.zeros_like(Lb)).sum(dim=2)  # (R,B)
        loss_in = (lse_term
                   - smoothing * sum_logits / total_classes
                   - (1 - smoothing) * logits_label.unsqueeze(1))
        loss_off = smoothing * (lse_term - sum_logits / total_classes)
        loss = torch.where(in_block, loss_in, loss_off)
    else:
        loss_in = lse_term - logits_label.unsqueeze(1)
        loss = torch.where(in_block, loss_in, torch.zeros_like(lse))

    if not SPLIT:
        z_loss = lse_square_scale * lse * lse                           # added to every non-ignored block
        loss = loss + z_loss
    else:
        z_loss = torch.zeros_like(lse)

    # Ignored rows -> loss=0, z_loss=0 (lse is still produced, matching the kernel,
    # which stores lse before the ignored-index check).
    ni = not_ignored.unsqueeze(1)
    loss = torch.where(ni, loss, torch.zeros_like(loss))
    z_loss = torch.where(ni, z_loss, torch.zeros_like(z_loss))

    # Assemble into the kernel's flat [col_block_idx * n_rows + row_idx] order, then
    # reshape/pad to the wrapper's [n_rows, n_cols] allocation (identical to before).
    def _layout(mat):  # mat: (R, B) -> stored (B, R) flat -> [n_rows, n_cols]
        flat = mat.transpose(0, 1).reshape(-1)
        total = n_rows * n_cols
        if flat.numel() < total:
            out = torch.zeros(total, dtype=torch.float32, device=dev)
            out[: flat.numel()] = flat
        else:
            out = flat[:total]
        return out.reshape(n_rows, n_cols)

    return _layout(loss), _layout(lse), _layout(z_loss)


def cross_entropy_bwd(
    dloss, logits, lse, labels, smoothing, logit_scale, lse_square_scale,
    ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING
):
    """
    PyTorch reference matching the Triton cross_entropy_bwd_kernel exactly.
    (Not exercised by test.py / benchmark.py — kept for API parity with the
    Triton/TileLang impls; left as the explicit reference loop.)
    """
    n_rows, n_cols = logits.shape
    num_col_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    dlogits = torch.empty_like(logits)

    for row_idx in range(n_rows):
        label_idx = labels[row_idx].item()

        if label_idx != ignored_index:
            dl = dloss[row_idx].item()
        else:
            dl = 0.0

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = torch.arange(BLOCK_SIZE, device=logits.device) + col_start
            mask = col_offsets < n_cols

            block_logits = torch.full((BLOCK_SIZE,), float('-inf'), dtype=torch.float32, device=logits.device)
            valid_cols = col_offsets[mask]
            block_logits[mask] = logits[row_idx, valid_cols].float() * logit_scale

            lse_val = lse[row_idx].float()
            probs = torch.exp(block_logits - lse_val)
            probs = probs + 2.0 * lse_square_scale * lse_val * probs

            adj_label = label_idx - class_start_idx
            if HAS_SMOOTHING:
                smooth_negative = smoothing / total_classes
                probs = torch.where(col_offsets == adj_label, probs - (1 - smoothing), probs) - smooth_negative
            else:
                probs = torch.where(col_offsets == adj_label, probs - 1.0, probs)

            result = (dl * logit_scale) * probs
            # Store valid columns
            for j in range(BLOCK_SIZE):
                col = col_start + j
                if col < n_cols:
                    dlogits[row_idx, col] = result[j]

    return dlogits
