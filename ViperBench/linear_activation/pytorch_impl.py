import torch
import torch.nn.functional as F


def kernel_ff(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor) -> torch.Tensor:
    """
    Llama-style feed-forward block matching Triton kernel precision:
      1. RMS normalize x using rms_w weights
      2. Dual matmul with w1 and w3
      3. SiLU gating: output = silu(normed_x @ w1.T) * (normed_x @ w3.T)

    The Triton kernel:
      - Multiplies a (fp16) * rms_w (fp16) -> stays fp16
      - Accumulates dot products in fp32
      - Computes a_sum = sum(a^2) in fp32
      - Normalizes acc1, acc2 by rsqrt(a_sum/K + eps)
      - Applies silu gating in fp32
      - Stores result (truncated to output dtype)
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

    # Matmuls: fp16 inputs, fp32 accumulation, then normalize by a_norm
    # Use float32 for the matmul to match tl.dot fp32 accumulation
    acc1 = (x_scaled.float() @ w1.float().T)  # [M, N] fp32
    acc2 = (x_scaled.float() @ w3.float().T)  # [M, N] fp32

    # Normalize
    acc1 = acc1 * a_norm.unsqueeze(1)
    acc2 = acc2 * a_norm.unsqueeze(1)

    # SiLU gating in fp32
    out = (acc1 * torch.sigmoid(acc1)) * acc2

    return out.half().view(batch, seq_len, N)
