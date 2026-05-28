import torch
import torch.nn.functional as F

def embedding(input_ids, weight, vob_start_id=0, vob_end_id=None, out=None):
    """
    Embedding lookup with vocabulary range masking.
    Looks up rows from weight table based on input_ids.
    """
    if vob_end_id is None:
        vob_end_id = weight.shape[0]
    valid = (input_ids >= vob_start_id) & (input_ids < vob_end_id)
    shifted_ids = input_ids - vob_start_id
    shifted_ids = shifted_ids.clamp(0, weight.shape[0] - 1)
    result = F.embedding(shifted_ids, weight)
    result = result * valid.unsqueeze(-1).float()
    if out is not None:
        out.copy_(result)
        return out
    return result
