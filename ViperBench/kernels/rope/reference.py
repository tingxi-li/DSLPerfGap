"""Reference: rope"""
import torch
import torch.nn.functional as F

def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    Q = inputs["Q"]
    K = inputs["K"]
    B, H, S, D = Q.shape
    position = torch.arange(S, device=Q.device).unsqueeze(1).float()
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=Q.device).float() / D))
    theta = position * freqs
    cos_t = torch.cos(theta).unsqueeze(0).unsqueeze(0)
    sin_t = torch.sin(theta).unsqueeze(0).unsqueeze(0)
    Q1, Q2 = Q[..., ::2], Q[..., 1::2]
    K1, K2 = K[..., ::2], K[..., 1::2]
    Q_rot = torch.stack([Q1 * cos_t - Q2 * sin_t, Q1 * sin_t + Q2 * cos_t], dim=-1).flatten(-2)
    K_rot = torch.stack([K1 * cos_t - K2 * sin_t, K1 * sin_t + K2 * cos_t], dim=-1).flatten(-2)
    return {"Q": Q_rot, "K": K_rot}
