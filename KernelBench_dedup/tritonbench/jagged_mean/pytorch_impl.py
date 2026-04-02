import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, values: torch.Tensor, offsets: torch.Tensor,
                M: int, max_seqlen: int) -> torch.Tensor:
        """
        Mean over variable-length rows using padded tensor approach.

        In a 3D nested tensor (B, *, M), computes the mean over the ragged
        dimension (*) for each of the B elements.

        Args:
            values: (total_elements, M) -- flattened jagged values
            offsets: (B+1,) -- cumulative row offsets
            M: inner dimension
            max_seqlen: maximum sequence length (for padding)

        Returns:
            output: (B, M) -- mean over ragged dimension per batch element
        """
        B = offsets.shape[0] - 1

        # Pad to dense (B, max_seqlen, M) tensor
        padded = torch.zeros(B, max_seqlen, M, device=values.device, dtype=values.dtype)
        for i in range(B):
            start = offsets[i].item()
            end = offsets[i + 1].item()
            length = end - start
            padded[i, :length] = values[start:end]

        # Sum and divide by actual lengths
        summed = padded.sum(dim=1)  # (B, M)
        lengths = (offsets[1:] - offsets[:-1]).float().unsqueeze(1).to(values.device)
        return summed / lengths


# Default shapes
B = 16
M = 64
MAX_SEQLEN = 128
DTYPE = torch.float32


def get_inputs():
    torch.manual_seed(42)
    lengths = torch.randint(1, MAX_SEQLEN + 1, (B,))
    offsets = torch.zeros(B + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    total = offsets[-1].item()
    values = torch.randn(total, M, dtype=DTYPE)
    max_seqlen = lengths.max().item()
    return [values, offsets, M, max_seqlen]


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
