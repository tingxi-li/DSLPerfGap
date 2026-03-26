import torch

def index_select(output, source, index):
    """
    Index-select rows from source by index.
    Equivalent to: output[:] = source[index]
    Supports ND source tensors: flattens to 2D (keeping dim 0), runs selection, reshapes back.
    Index must be 1D.
    """
    if not index.ndim == 1:
        raise ValueError(f"index must be 1D, got {index.ndim}D tensor with shape {index.shape}")

    orig_shape = source.shape
    if source.ndim > 2:
        source_2d = source.reshape(orig_shape[0], -1)
    elif source.ndim == 1:
        source_2d = source.unsqueeze(1)
    else:
        source_2d = source

    result = torch.index_select(source_2d, 0, index)

    if source.ndim > 2:
        out_shape = (index.shape[0],) + orig_shape[1:]
        result = result.reshape(out_shape)
    elif source.ndim == 1:
        result = result.squeeze(1)

    output.copy_(result)
    return output
