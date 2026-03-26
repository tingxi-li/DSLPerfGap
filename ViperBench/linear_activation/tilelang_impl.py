import tilelang
import tilelang.language as T
import torch


def _tilelang_gemm(A, B, M, N, K):
    """Compute C = A @ B using TileLang GEMM. A is [M,K] fp16, B is [K,N] fp16, C is [M,N] fp16."""
    block_M, block_N, block_K = 128, 128, 32

    M_pad = ((M + block_M - 1) // block_M) * block_M
    N_pad = ((N + block_N - 1) // block_N) * block_N
    K_pad = ((K + block_K - 1) // block_K) * block_K

    @tilelang.jit
    def kernel(m, n, k, bM=block_M, bN=block_N, bK=block_K):
        @T.prim_func
        def func(
            A_in: T.Tensor((m, k), "float16"),
            B_in: T.Tensor((k, n), "float16"),
            C_out: T.Tensor((m, n), "float16"),
        ):
            with T.Kernel(T.ceildiv(n, bN), T.ceildiv(m, bM), threads=128) as (bx, by):
                A_shared = T.alloc_shared((bM, bK), "float16")
                B_shared = T.alloc_shared((bK, bN), "float16")
                C_local = T.alloc_fragment((bM, bN), "float32")
                T.use_swizzle(panel_size=10)
                T.clear(C_local)
                for ki in T.Pipelined(T.ceildiv(k, bK), num_stages=3):
                    T.copy(A_in[by * bM, ki * bK], A_shared)
                    T.copy(B_in[ki * bK, bx * bN], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_out[by * bM, bx * bN])
        return func

    # Pad inputs if needed
    if M_pad != M or K_pad != K:
        a_pad = torch.zeros(M_pad, K_pad, device=A.device, dtype=torch.float16)
        a_pad[:M, :K] = A
    else:
        a_pad = A.contiguous()

    if K_pad != K or N_pad != N:
        b_pad = torch.zeros(K_pad, N_pad, device=B.device, dtype=torch.float16)
        b_pad[:K, :N] = B
    else:
        b_pad = B.contiguous()

    c_pad = torch.zeros(M_pad, N_pad, device=A.device, dtype=torch.float16)
    func = kernel(M_pad, N_pad, K_pad)
    func(a_pad, b_pad, c_pad)
    return c_pad[:M, :N]


def kernel_ff(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor) -> torch.Tensor:
    """
    Llama-style feed-forward block using TileLang for GEMMs:
      1. RMS normalize x using rms_w weights
      2. Dual matmul with w1 and w3 (via TileLang)
      3. SiLU gating: output = silu(normed_x @ w1.T) * (normed_x @ w3.T)
    """
    x = x.half() if x.dtype != torch.float16 else x
    w1 = w1.half() if w1.dtype != torch.float16 else w1
    w3 = w3.half() if w3.dtype != torch.float16 else w3
    rms_w = rms_w.half() if rms_w.dtype != torch.float16 else rms_w

    batch, seq_len, dim = x.shape
    M = batch * seq_len
    K = dim
    N = w1.shape[0]

    x_flat = x.reshape(M, K)  # [M, K] fp16

    # Compute a_sum in float32 (matching Triton's pow(a.to(float32), 2))
    eps = 1e-6
    x_f32 = x_flat.float()
    a_sq_sum = (x_f32 ** 2).sum(dim=-1)  # [M]
    a_norm = torch.rsqrt(a_sq_sum / K + eps)  # [M]

    # x * rms_w in fp16 (matching Triton: a = a * rms_w, both fp16)
    x_scaled = x_flat * rms_w.unsqueeze(0)  # [M, K] fp16

    # Use TileLang GEMMs: x_scaled @ w1.T and x_scaled @ w3.T
    # w1 is [N, K], w1.T is [K, N]
    w1_t = w1.t().contiguous()  # [K, N]
    w3_t = w3.t().contiguous()  # [K, N]

    acc1 = _tilelang_gemm(x_scaled, w1_t, M, N, K)  # [M, N] fp16
    acc2 = _tilelang_gemm(x_scaled, w3_t, M, N, K)  # [M, N] fp16

    # Convert to fp32 for normalization and activation
    acc1_f32 = acc1.float()
    acc2_f32 = acc2.float()

    # Normalize by a_norm
    acc1_f32 = acc1_f32 * a_norm.unsqueeze(1)
    acc2_f32 = acc2_f32 * a_norm.unsqueeze(1)

    # SiLU gating in fp32
    out = (acc1_f32 * torch.sigmoid(acc1_f32)) * acc2_f32

    return out.half().view(batch, seq_len, N)
