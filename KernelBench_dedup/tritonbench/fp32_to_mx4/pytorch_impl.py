import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, group_size: int, ebits: int,
                mbits: int) -> torch.Tensor:
        """
        Pure PyTorch approximation of fp32 to mx4 quantization.

        MX4 format: each element is stored as a 4-bit value (2 exponent bits + 1 mantissa bit + 1 sign bit),
        with a shared scale per group. This packs 2 elements per byte plus 1 scale byte per group.

        Args:
            x: (N,) float32 input tensor
            group_size: number of elements per group (typically 32)
            ebits: exponent bits (typically 2)
            mbits: mantissa bits (typically 1)

        Returns:
            packed: (N//2 + N//group_size,) uint8 packed tensor
        """
        N = x.numel()
        x_flat = x.reshape(-1)

        # Pad to multiple of group_size
        pad_size = (group_size - N % group_size) % group_size
        if pad_size > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_size))

        N_padded = x_flat.numel()
        num_groups = N_padded // group_size

        # Reshape into groups
        x_groups = x_flat.reshape(num_groups, group_size)

        # Compute shared exponent (scale) per group: max absolute value
        group_max = x_groups.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)

        # Quantize: scale to [-1, 1] range, then map to 4-bit integers
        # With ebits=2, mbits=1: we have 4 levels positive and 4 levels negative
        max_val = (2 ** (2 ** ebits - 1)) * (2 - 2 ** (-mbits))
        scaled = x_groups / group_max * max_val

        # Round to nearest representable value (simple round-to-nearest)
        # For mx4 with sign+2exp+1man, representable positive values are: 0, 0.5, 1, 1.5, 2, 3, 4, 6
        clamp_max = 2 ** (2 ** ebits - 1)
        quantized = scaled.clamp(-clamp_max, clamp_max).round().to(torch.int8)

        # Pack two 4-bit values per byte
        quantized_unsigned = (quantized + clamp_max).clamp(0, 15).to(torch.uint8)
        even = quantized_unsigned[:, 0::2]
        odd = quantized_unsigned[:, 1::2]
        packed_data = (even & 0x0F) | ((odd & 0x0F) << 4)

        # Compute scale bytes (simplified: quantize log2 of group_max to uint8)
        log_scale = torch.log2(group_max.squeeze(1)).clamp(-127, 127)
        scale_bytes = ((log_scale + 127).round()).to(torch.uint8)

        # Concatenate packed data and scales
        packed_flat = packed_data.reshape(-1)
        output = torch.cat([packed_flat, scale_bytes.reshape(-1)])

        return output


# Default shapes from operator.py
SIZE = 1024 * 1024
GROUP_SIZE = 32
EBITS = 2
MBITS = 1


def get_inputs():
    x = torch.randn(SIZE, dtype=torch.float32)
    return [x, GROUP_SIZE, EBITS, MBITS]


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
