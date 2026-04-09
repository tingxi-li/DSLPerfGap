"""
TileLang implementation for: fp32_to_mx4
Status: NOT_IMPLEMENTED
"""
import torch

NOT_IMPLEMENTED = True
REASON = "Awaiting implementation"


def kernel(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    raise NotImplementedError(REASON)
