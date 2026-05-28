import torch


def matrix_transpose(x):
    """Unified API: matrix_transpose(x) -> Tensor, returns x.T contiguous.
    Only supports 2D input matrices.
    """
    if x.ndim != 2:
        raise ValueError(f"Only 2D matrices supported, got {x.ndim}D tensor with shape {x.shape}")
    return x.T.contiguous()
