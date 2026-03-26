import torch


def attention_fwd(q, k, v, store=False, ifcond=False):
    """
    Chunked linear attention with state accumulation.

    Args:
        q, k, v: [B, H, T, D] float32
        store: whether to store intermediate state buffer h
        ifcond: if True, skip cross-block contribution for first block

    Returns:
        o: [B, H, T, D] same dtype as q
        (h is stored internally when store=True, matching Triton behavior)
    """
    B, H, T, D = q.shape
    BT = 32
    NT = (T + BT - 1) // BT
    scale = D ** -0.5

    o = torch.empty_like(q)
    # Allocate state buffer matching Triton's layout: [B, H, NT*D, D]
    if store:
        h = q.new_empty(B, H, NT * D, D)

    for b in range(B):
        for hh in range(H):
            # State buffer [D, D]
            b_h = torch.zeros(D, D, device=q.device, dtype=torch.float32)

            num_blocks = NT
            for i in range(num_blocks):
                start = i * BT
                end = min(start + BT, T)

                # Store state BEFORE processing this block (matches Triton)
                if store:
                    h[b, hh, i * D:(i + 1) * D, :] = b_h.to(q.dtype)

                b_q = q[b, hh, start:end, :].float() * scale  # [BT, D]
                b_k = k[b, hh, start:end, :].float()           # [BT, D]
                b_v = v[b, hh, start:end, :].float()           # [BT, D]

                # Intra-block: b_s = q @ k^T, b_o = b_s @ v
                b_s = b_q @ b_k.T                             # [BT, BT]
                b_o = b_s.to(b_q.dtype) @ b_v                 # [BT, D]

                if ifcond:
                    if i == 0:
                        b_h = b_k.T @ b_v                     # [D, D]
                    else:
                        b_o = b_o + b_q @ b_h.to(b_q.dtype)   # cross-block
                        b_h = b_h + b_k.T @ b_v               # update state
                else:
                    b_o = b_o + b_q @ b_h.to(b_q.dtype)       # cross-block
                    b_h = b_h + b_k.T @ b_v                   # update state

                o[b, hh, start:end, :] = b_o.to(q.dtype)

    return o
