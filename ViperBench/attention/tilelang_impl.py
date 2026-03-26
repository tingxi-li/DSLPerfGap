"""
Chunked linear attention using TileLang @tilelang.jit kernels.

Uses element-wise multiply-accumulate (not T.gemm) for full fp32 precision.
T.gemm uses TF32 tensor cores which only have 10-bit mantissa, insufficient
for the float32 tolerance (atol=2e-5) required by this kernel.

Algorithm (per batch, head):
  For each chunk i of BT=32 tokens:
    1. Intra-block: b_s = (q_chunk * scale) @ k_chunk^T,  b_o = b_s @ v_chunk
    2. Cross-block: b_o += q_chunk @ b_h
    3. State update: b_h += k_chunk^T @ v_chunk
"""
import torch
import tilelang
import tilelang.language as T

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("attention", "tilelang") or {}
except Exception:
    _TUNED = {}

_attn_threads = _TUNED.get("threads", 128)
_attn_block_M = _TUNED.get("block_M", 32)
_attn_block_N = _TUNED.get("block_N", 32)
_attn_block_K = _TUNED.get("block_K", 32)


def _make_matmul_kernel(M, N, K, block_M=_attn_block_M, block_N=_attn_block_N, block_K=_attn_block_K):
    """
    Create a TileLang matmul kernel: C[M,N] = A[M,K] @ B[K,N]
    Uses element-wise multiply-accumulate for full fp32 precision.
    """
    @tilelang.jit
    def kernel(M_dim, N_dim, K_dim, bM=block_M, bN=block_N, bK=block_K):
        @T.prim_func
        def func(
            A_t: T.Tensor((M_dim, K_dim), "float32"),
            B_t: T.Tensor((K_dim, N_dim), "float32"),
            C_t: T.Tensor((M_dim, N_dim), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(N_dim, bN), T.ceildiv(M_dim, bM),
                threads=_attn_threads
            ) as (bx, by):
                C_local = T.alloc_fragment((bM, bN), "float32")
                A_local = T.alloc_fragment((bM,), "float32")
                B_local = T.alloc_fragment((bN,), "float32")

                T.clear(C_local)

                for k_tile in range(T.ceildiv(K_dim, bK)):
                    for kk in range(bK):
                        k_idx = k_tile * bK + kk
                        for i in T.Parallel(bM):
                            A_local[i] = A_t[by * bM + i, k_idx]
                        for j in T.Parallel(bN):
                            B_local[j] = B_t[k_idx, bx * bN + j]
                        for i, j in T.Parallel(bM, bN):
                            C_local[i, j] += A_local[i] * B_local[j]

                for i, j in T.Parallel(bM, bN):
                    C_t[by * bM + i, bx * bN + j] = C_local[i, j]

        return func

    return kernel(M, N, K)


def _make_matmul_add_kernel(M, N, K, block_M=_attn_block_M, block_N=_attn_block_N, block_K=_attn_block_K):
    """
    Create a TileLang kernel: C[M,N] += A[M,K] @ B[K,N]
    Reads existing C values and adds to them.
    """
    @tilelang.jit
    def kernel(M_dim, N_dim, K_dim, bM=block_M, bN=block_N, bK=block_K):
        @T.prim_func
        def func(
            A_t: T.Tensor((M_dim, K_dim), "float32"),
            B_t: T.Tensor((K_dim, N_dim), "float32"),
            C_t: T.Tensor((M_dim, N_dim), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(N_dim, bN), T.ceildiv(M_dim, bM),
                threads=_attn_threads
            ) as (bx, by):
                C_local = T.alloc_fragment((bM, bN), "float32")
                A_local = T.alloc_fragment((bM,), "float32")
                B_local = T.alloc_fragment((bN,), "float32")

                # Load existing C values
                for i, j in T.Parallel(bM, bN):
                    C_local[i, j] = C_t[by * bM + i, bx * bN + j]

                for k_tile in range(T.ceildiv(K_dim, bK)):
                    for kk in range(bK):
                        k_idx = k_tile * bK + kk
                        for i in T.Parallel(bM):
                            A_local[i] = A_t[by * bM + i, k_idx]
                        for j in T.Parallel(bN):
                            B_local[j] = B_t[k_idx, bx * bN + j]
                        for i, j in T.Parallel(bM, bN):
                            C_local[i, j] += A_local[i] * B_local[j]

                for i, j in T.Parallel(bM, bN):
                    C_t[by * bM + i, bx * bN + j] = C_local[i, j]

        return func

    return kernel(M, N, K)


# Cache for compiled kernels keyed by (M, N, K)
_kernel_cache = {}


def _get_matmul_kernel(M, N, K):
    key = ("matmul", M, N, K)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_matmul_kernel(M, N, K)
    return _kernel_cache[key]


def _get_matmul_add_kernel(M, N, K):
    key = ("matmul_add", M, N, K)
    if key not in _kernel_cache:
        _kernel_cache[key] = _make_matmul_add_kernel(M, N, K)
    return _kernel_cache[key]


def _tl_matmul(A, B, device):
    """Compute A @ B using TileLang kernel. A: [M,K], B: [K,N] -> [M,N]"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.zeros(M, N, device=device, dtype=torch.float32)
    kern = _get_matmul_kernel(M, N, K)
    kern(A.contiguous(), B.contiguous(), C)
    return C


def _tl_matmul_add(A, B, C, device):
    """Compute C += A @ B using TileLang kernel. A: [M,K], B: [K,N], C: [M,N]"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert C.shape == (M, N)
    kern = _get_matmul_add_kernel(M, N, K)
    kern(A.contiguous(), B.contiguous(), C)
    return C


def attention_fwd(q, k, v, store=False, ifcond=False):
    """
    Chunked linear attention with state accumulation using TileLang kernels.

    Args:
        q, k, v: [B, H, T, D] float32
        store: whether to store intermediate state buffer h
        ifcond: if True, skip cross-block contribution for first block

    Returns:
        o: [B, H, T, D] same dtype as q
    """
    B, H, T_len, D = q.shape
    BT = 32
    NT = (T_len + BT - 1) // BT
    scale = D ** -0.5
    device = q.device

    o = torch.empty_like(q)
    qf = q.float()
    kf = k.float()
    vf = v.float()

    for b in range(B):
        for hh in range(H):
            b_h = torch.zeros(D, D, device=device, dtype=torch.float32)

            for i in range(NT):
                start = i * BT
                end = min(start + BT, T_len)
                chunk = end - start

                b_q = qf[b, hh, start:end, :] * scale  # [chunk, D]
                b_k = kf[b, hh, start:end, :]           # [chunk, D]
                b_v = vf[b, hh, start:end, :]           # [chunk, D]

                # Pad chunk to BT if needed (last chunk may be smaller)
                if chunk < BT:
                    pad_q = torch.zeros(BT, D, device=device, dtype=torch.float32)
                    pad_k = torch.zeros(BT, D, device=device, dtype=torch.float32)
                    pad_v = torch.zeros(BT, D, device=device, dtype=torch.float32)
                    pad_q[:chunk] = b_q
                    pad_k[:chunk] = b_k
                    pad_v[:chunk] = b_v
                    b_q_p, b_k_p, b_v_p = pad_q, pad_k, pad_v
                else:
                    b_q_p, b_k_p, b_v_p = b_q, b_k, b_v

                # Intra-block: b_s = b_q @ b_k^T  [BT, BT]
                b_s = _tl_matmul(b_q_p, b_k_p.T.contiguous(), device)

                # b_o = b_s @ b_v  [BT, D]
                b_o = _tl_matmul(b_s, b_v_p, device)

                if ifcond:
                    if i == 0:
                        # b_h = b_k^T @ b_v  [D, D]
                        b_h = _tl_matmul(b_k_p.T.contiguous(), b_v_p, device)
                    else:
                        # b_o += b_q @ b_h
                        _tl_matmul_add(b_q_p, b_h, b_o, device)
                        # b_h += b_k^T @ b_v
                        _tl_matmul_add(b_k_p.T.contiguous(), b_v_p, b_h, device)
                else:
                    # b_o += b_q @ b_h
                    _tl_matmul_add(b_q_p, b_h, b_o, device)
                    # b_h += b_k^T @ b_v
                    _tl_matmul_add(b_k_p.T.contiguous(), b_v_p, b_h, device)

                o[b, hh, start:end, :] = b_o[:chunk].to(q.dtype)

    return o
