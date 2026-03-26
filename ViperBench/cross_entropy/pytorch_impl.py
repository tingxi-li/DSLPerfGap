import torch


def cross_entropy_fwd(
    logits, labels, smoothing, logit_scale, lse_square_scale,
    ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING, SPLIT
):
    """
    PyTorch reference matching the Triton cross_entropy_fwd_kernel exactly.

    The Triton kernel processes per (row, col_block) and stores results in a flat buffer
    indexed as [col_block_idx * n_rows + row_idx].

    Returns (loss, lse, z_loss) each of shape matching the Triton output layout.
    """
    n_rows, n_cols = logits.shape
    num_col_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Output buffers: flat [num_col_blocks * n_rows], but we'll reshape to [num_col_blocks, n_rows]
    # to match Triton's store pattern: loss_ptr + col_block_idx * n_rows + row_idx
    loss_out = torch.empty(num_col_blocks, n_rows, dtype=torch.float32, device=logits.device)
    lse_out = torch.empty(num_col_blocks, n_rows, dtype=torch.float32, device=logits.device)
    z_loss_out = torch.empty(num_col_blocks, n_rows, dtype=torch.float32, device=logits.device)

    for row_idx in range(n_rows):
        label_idx = labels[row_idx].item()

        for col_block_idx in range(num_col_blocks):
            col_start = col_block_idx * BLOCK_SIZE
            col_offsets = torch.arange(BLOCK_SIZE, device=logits.device) + col_start
            mask = col_offsets < n_cols

            # Load logits with masking (out-of-bounds -> -inf)
            block_logits = torch.full((BLOCK_SIZE,), float('-inf'), dtype=torch.float32, device=logits.device)
            valid_cols = col_offsets[mask]
            block_logits[mask] = logits[row_idx, valid_cols].float() * logit_scale

            max_logits = block_logits.max()

            if HAS_SMOOTHING:
                sum_logits = torch.where(mask, block_logits, torch.zeros_like(block_logits)).sum()

            lse = torch.log(torch.exp(block_logits - max_logits).sum()) + max_logits

            lse_out[col_block_idx, row_idx] = lse

            if label_idx == ignored_index:
                loss = 0.0
                z_loss = 0.0
            else:
                adj_label = label_idx - class_start_idx
                label_in_block = (adj_label >= col_start) and (adj_label < min(n_cols, col_start + BLOCK_SIZE))

                if label_in_block:
                    logits_label = logits[row_idx, adj_label].float() * logit_scale
                    if HAS_SMOOTHING:
                        loss = (
                            (lse.item() if not SPLIT else 0.0)
                            - smoothing * sum_logits.item() / total_classes
                            - (1 - smoothing) * logits_label.item()
                        )
                    else:
                        loss = (lse.item() if not SPLIT else 0.0) - logits_label.item()
                else:
                    if HAS_SMOOTHING:
                        loss = smoothing * ((lse.item() if not SPLIT else 0.0) - sum_logits.item() / total_classes)
                    else:
                        loss = 0.0

                if not SPLIT:
                    z_loss = lse_square_scale * lse.item() * lse.item()
                    loss += z_loss
                else:
                    z_loss = 0.0

            loss_out[col_block_idx, row_idx] = loss
            if not SPLIT:
                z_loss_out[col_block_idx, row_idx] = z_loss

    # The Triton wrapper allocates [n_rows, n_cols] and the kernel writes to
    # flat offset col_block_idx * n_rows + row_idx. We need to match that exact memory layout.
    # Reshape to match the wrapper's expected [n_rows, n_cols] allocation.
    # The flat buffer has num_col_blocks * n_rows elements stored contiguously.
    loss_flat = loss_out.reshape(-1)
    lse_flat = lse_out.reshape(-1)
    z_loss_flat = z_loss_out.reshape(-1)

    # Pad to match [n_rows, n_cols] if needed
    total_alloc = n_rows * n_cols
    if loss_flat.numel() < total_alloc:
        loss_final = torch.zeros(total_alloc, dtype=torch.float32, device=logits.device)
        lse_final = torch.zeros(total_alloc, dtype=torch.float32, device=logits.device)
        z_loss_final = torch.zeros(total_alloc, dtype=torch.float32, device=logits.device)
        loss_final[:loss_flat.numel()] = loss_flat
        lse_final[:lse_flat.numel()] = lse_flat
        z_loss_final[:z_loss_flat.numel()] = z_loss_flat
        loss_final = loss_final.reshape(n_rows, n_cols)
        lse_final = lse_final.reshape(n_rows, n_cols)
        z_loss_final = z_loss_final.reshape(n_rows, n_cols)
    else:
        loss_final = loss_flat[:total_alloc].reshape(n_rows, n_cols)
        lse_final = lse_flat[:total_alloc].reshape(n_rows, n_cols)
        z_loss_final = z_loss_flat[:total_alloc].reshape(n_rows, n_cols)

    return loss_final, lse_final, z_loss_final


def cross_entropy_bwd(
    dloss, logits, lse, labels, smoothing, logit_scale, lse_square_scale,
    ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING
):
    """
    PyTorch reference matching the Triton cross_entropy_bwd_kernel exactly.
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
