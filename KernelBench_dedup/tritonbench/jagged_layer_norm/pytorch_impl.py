import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = 1e-6


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, values: torch.Tensor, offsets: torch.Tensor,
                M: int) -> torch.Tensor:
        """
        Layer normalization on jagged (variable-length) rows.
        Uses padded tensor approach: pad to max length, normalize, then extract.

        Args:
            values: (total_elements, M) -- flattened jagged values
            offsets: (B+1,) -- cumulative row offsets
            M: inner dimension size

        Returns:
            output: (total_elements, M) -- layer-normalized values
        """
        B = offsets.shape[0] - 1
        max_seqlen = 0
        for i in range(B):
            length = (offsets[i + 1] - offsets[i]).item()
            if length > max_seqlen:
                max_seqlen = length

        # Pad to (B, max_seqlen, M) dense tensor
        padded = torch.zeros(B, max_seqlen, M, device=values.device, dtype=values.dtype)
        mask = torch.zeros(B, max_seqlen, 1, device=values.device, dtype=values.dtype)

        for i in range(B):
            start = offsets[i].item()
            end = offsets[i + 1].item()
            length = end - start
            padded[i, :length] = values[start:end]
            mask[i, :length] = 1.0

        # Compute mean and variance per batch element over (seq, M)
        ragged_lengths = (offsets[1:] - offsets[:-1]).float().unsqueeze(1).unsqueeze(2) * M
        mean = (padded * mask).sum(dim=(1, 2), keepdim=True) / ragged_lengths
        normalized = (padded - mean) * mask
        variance = (normalized ** 2).sum(dim=(1, 2), keepdim=True) / ragged_lengths
        padded_ln = normalized / torch.sqrt(variance + EPSILON)

        # Extract back to jagged format
        output = torch.zeros_like(values)
        for i in range(B):
            start = offsets[i].item()
            end = offsets[i + 1].item()
            length = end - start
            output[start:end] = padded_ln[i, :length]

        return output


# Default shapes
B = 16
M = 64
MAX_SEQLEN = 128
DTYPE = torch.float32


def get_inputs():
    # Create simple jagged data: each batch element has a random length
    torch.manual_seed(42)
    lengths = torch.randint(1, MAX_SEQLEN + 1, (B,))
    offsets = torch.zeros(B + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    total = offsets[-1].item()
    values = torch.randn(total, M, dtype=DTYPE)
    return [values, offsets, M]


def get_init_inputs():
    return []


def get_test_inputs():
    return [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
