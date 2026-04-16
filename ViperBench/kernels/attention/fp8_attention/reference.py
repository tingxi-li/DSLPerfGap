"""Reference: fp8_attention"""
import torch
from torchao.prototype.attention.fp8_fa3 import fp8_fa3_sdpa


def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    Q = inputs["Q"]
    K = inputs["K"]
    V = inputs["V"]

    # Basic input validation
    if not (Q.is_cuda and K.is_cuda and V.is_cuda):
        raise RuntimeError("fp8_fa3_sdpa requires CUDA tensors.")
    if Q.device != K.device or Q.device != V.device:
        raise RuntimeError(
            f"Q, K, V must be on the same device, got {Q.device}, {K.device}, {V.device}."
        )
    if not (Q.dtype == K.dtype == V.dtype):
        raise RuntimeError(
            f"Q, K, V must have the same dtype, got {Q.dtype}, {K.dtype}, {V.dtype}."
        )
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise RuntimeError(
            f"Expected Q, K, V to have shape [B, H, S, D], got "
            f"{tuple(Q.shape)}, {tuple(K.shape)}, {tuple(V.shape)}."
        )

    # Shape consistency checks
    if Q.shape[0] != K.shape[0] or Q.shape[0] != V.shape[0]:
        raise RuntimeError("Batch size of Q, K, V must match.")
    if Q.shape[1] != K.shape[1] or Q.shape[1] != V.shape[1]:
        raise RuntimeError("Number of heads of Q, K, V must match.")
    if K.shape[2] != V.shape[2]:
        raise RuntimeError("Sequence length of K and V must match.")
    if K.shape[3] != V.shape[3]:
        raise RuntimeError("Head dimension of K and V must match.")
    if Q.shape[3] != K.shape[3]:
        raise RuntimeError("Head dimension of Q and K must match.")

    out = fp8_fa3_sdpa(Q, K, V)
    return {"output": out}