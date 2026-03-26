import tilelang
import tilelang.language as T
import torch


def embedding(input_ids, weight, vob_start_id=0, vob_end_id=None, out=None):
    """
    TileLang embedding lookup with vocabulary range masking.
    input_ids: (N,) int32
    weight: (V, D) float32
    Returns: (N, D) float32
    """
    if vob_end_id is None:
        vob_end_id = weight.shape[0]

    N = input_ids.shape[0]
    orig_dtype = weight.dtype
    V, D = weight.shape

    # Pre-compute shifted ids and validity mask on PyTorch side
    valid = (input_ids >= vob_start_id) & (input_ids < vob_end_id)
    shifted_ids = (input_ids - vob_start_id).clamp(0, V - 1).to(torch.int32)
    shifted_ids = shifted_ids * valid.int()
    valid_mask = valid.float()
    weight = weight.float()

    block_N = 4
    block_D = min(128, D)

    # Pad D to multiple of block_D
    D_pad = ((D + block_D - 1) // block_D) * block_D
    N_pad = ((N + block_N - 1) // block_N) * block_N

    if D_pad != D:
        weight_pad = torch.zeros(V, D_pad, device=weight.device, dtype=weight.dtype)
        weight_pad[:, :D] = weight
    else:
        weight_pad = weight

    if N_pad != N:
        ids_pad = torch.zeros(N_pad, device=input_ids.device, dtype=torch.int32)
        ids_pad[:N] = shifted_ids
        mask_pad = torch.zeros(N_pad, device=weight.device, dtype=torch.float32)
        mask_pad[:N] = valid_mask
    else:
        ids_pad = shifted_ids
        mask_pad = valid_mask

    if out is not None:
        out_pad = torch.zeros(N_pad, D_pad, device=weight.device, dtype=weight.dtype)
    else:
        out_pad = torch.zeros(N_pad, D_pad, device=weight.device, dtype=weight.dtype)

    @tilelang.jit
    def kernel(n_size, v_size, d_size, bN=block_N, bD=block_D):
        @T.prim_func
        def func(
            ids: T.Tensor((n_size,), "int32"),
            mask: T.Tensor((n_size,), "float32"),
            W: T.Tensor((v_size, d_size), "float32"),
            Out: T.Tensor((n_size, d_size), "float32"),
        ):
            with T.Kernel(T.ceildiv(d_size, bD), T.ceildiv(n_size, bN), threads=128) as (bx, by):
                out_local = T.alloc_fragment((bN, bD), "float32")
                T.clear(out_local)
                for i, j in T.Parallel(bN, bD):
                    row = by * bN + i
                    tok_idx = ids[row]
                    m = mask[row]
                    out_local[i, j] = W[tok_idx, bx * bD + j] * m
                T.copy(out_local, Out[by * bN, bx * bD])
        return func

    k = kernel(N_pad, V, D_pad)
    k(ids_pad, mask_pad, weight_pad, out_pad)

    result = out_pad[:N, :D].to(orig_dtype).contiguous()
    if out is not None:
        out.copy_(result)
        return out
    return result
