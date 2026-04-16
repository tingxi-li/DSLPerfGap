"""
Triton implementation for: huber_loss
Status: NOT_IMPLEMENTED
"""
import torch

NOT_IMPLEMENTED = True
REASON = "Awaiting implementation"


def kernel(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    raise NotImplementedError(REASON)
