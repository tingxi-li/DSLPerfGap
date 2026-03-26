import torch
import tilelang
import tilelang.language as T
import math


def argmax(input_tensor, dim, keepdim=False):
    """
    Returns the indices of the maximum values across a specified dimension.
    Uses TileLang primitives for the core computation.
    """
    inp = input_tensor
    assert dim is not None, "dim must be specified"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim]) if dim > 0 else 1
    K = inp.numel() // M // N

    inp_contig = inp.contiguous().float()

    # Reshape to (M, N, K) for uniform handling
    inp_3d = inp_contig.view(M, N, K)

    # For each (m, k), find argmax over n
    block_M = min(32, M) if M > 0 else 1
    block_K = min(32, K) if K > 0 else 1

    # Pad M and K to multiples of block sizes
    M_pad = ((M + block_M - 1) // block_M) * block_M
    K_pad = ((K + block_K - 1) // block_K) * block_K

    if M_pad != M or K_pad != K:
        inp_pad = torch.full((M_pad, N, K_pad), float('-inf'), device=inp.device, dtype=torch.float32)
        inp_pad[:M, :, :K] = inp_3d
        out_pad = torch.zeros(M_pad, K_pad, dtype=torch.int32, device=inp.device)
    else:
        inp_pad = inp_3d
        out_pad = torch.empty(M, K, dtype=torch.int32, device=inp.device)

    @tilelang.jit
    def argmax_kernel(m_size, n_size, k_size, bM=block_M, bK=block_K):
        @T.prim_func
        def func(
            A: T.Tensor((m_size, n_size, k_size), "float32"),
            Out: T.Tensor((m_size, k_size), "int32"),
        ):
            with T.Kernel(T.ceildiv(k_size, bK), T.ceildiv(m_size, bM), threads=128) as (bx, by):
                max_val = T.alloc_fragment((bM, bK), "float32")
                max_idx = T.alloc_fragment((bM, bK), "int32")
                cur_val = T.alloc_fragment((bM, bK), "float32")
                T.clear(max_idx)
                # Initialize max_val to -inf
                for i, j in T.Parallel(bM, bK):
                    max_val[i, j] = T.float32(-1e30)
                for n in T.serial(n_size):
                    for i, j in T.Parallel(bM, bK):
                        cur_val[i, j] = A[by * bM + i, n, bx * bK + j]
                    for i, j in T.Parallel(bM, bK):
                        if cur_val[i, j] > max_val[i, j]:
                            max_val[i, j] = cur_val[i, j]
                            max_idx[i, j] = T.int32(n)
                T.copy(max_idx, Out[by * bM, bx * bK])
        return func

    kernel = argmax_kernel(M_pad, N, K_pad)
    kernel(inp_pad, out_pad)

    if M_pad != M or K_pad != K:
        out_flat = out_pad[:M, :K].to(torch.int64)
    else:
        out_flat = out_pad.to(torch.int64)

    # Reshape output
    shape_list = list(shape)
    shape_list[dim] = 1
    result = out_flat.view(shape_list)
    if not keepdim:
        result = result.squeeze(dim)
    return result
