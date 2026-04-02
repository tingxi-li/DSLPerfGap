import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, packed: torch.Tensor, group_size: int, ebits: int,
                mbits: int) -> torch.Tensor:
        """
        Pure PyTorch approximation of mx4 to fp32 dequantization.

        MX4 format stores packed 4-bit quantized values with shared group scales.
        This function reverses the packing.

        Args:
            packed: (M,) uint8 packed tensor containing data + scale bytes
            group_size: number of elements per group (typically 32)
            ebits: exponent bits (typically 2)
            mbits: mantissa bits (typically 1)

        Returns:
            output: (N,) float32 dequantized tensor
        """
        packed_group_size = group_size // 2 + 1  # data bytes + 1 scale byte per group
        total = packed.numel()
        num_groups = total // packed_group_size
        N = num_groups * group_size

        clamp_max = 2 ** (2 ** ebits - 1)

        # Split into groups
        packed_groups = packed.reshape(num_groups, packed_group_size)

        # Extract scale byte (last byte of each group)
        scale_bytes = packed_groups[:, -1].to(torch.float32)
        log_scale = (scale_bytes - 127.0).clamp(-126.0, 127.0)
        group_scale = torch.pow(2.0, log_scale)  # (num_groups,)

        # Extract packed data (first group_size//2 bytes)
        data_bytes = packed_groups[:, :group_size // 2]  # (num_groups, group_size//2)

        # Unpack two 4-bit values per byte
        low_nibble = (data_bytes & 0x0F).to(torch.float32)
        high_nibble = ((data_bytes >> 4) & 0x0F).to(torch.float32)

        # Interleave: even positions get low nibble, odd get high nibble
        unpacked = torch.stack([low_nibble, high_nibble], dim=-1)
        unpacked = unpacked.reshape(num_groups, group_size)  # (num_groups, group_size)

        # Convert back from unsigned to signed
        unpacked = unpacked - clamp_max

        # Dequantize: multiply by scale and normalize
        max_val = (2 ** (2 ** ebits - 1)) * (2 - 2 ** (-mbits))
        output = unpacked / max_val * group_scale.unsqueeze(1)

        return output.reshape(-1).to(torch.float32)


# Default shapes from operator.py
SIZE = 512 * 1024
GROUP_SIZE = 32
EBITS = 2
MBITS = 1


def get_inputs():
    # Create a simple packed mx4 tensor for testing
    # Simulate: quantize random fp32 data, then dequantize
    torch.manual_seed(42)
    packed_group_size = GROUP_SIZE // 2 + 1  # 17 bytes per group
    num_groups = SIZE // GROUP_SIZE
    total_packed = num_groups * packed_group_size

    # Create random packed data
    packed = torch.randint(0, 256, (total_packed,), dtype=torch.uint8)
    return [packed, GROUP_SIZE, EBITS, MBITS]


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
