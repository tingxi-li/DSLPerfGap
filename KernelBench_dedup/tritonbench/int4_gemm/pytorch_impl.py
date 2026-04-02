import torch
import torch.nn as nn


class Model(nn.Module):
    """INT4 GEMM baseline: unpack int4 weights to bf16 and matmul.

    The real operator packs two int4 values per byte, then uses a Triton kernel
    for the GEMM. This adapter unpacks int32 weights to bf16 and performs
    a standard matmul.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        # x: (B, L, Din) bf16 activation
        # w: (Din, Dout) int32 weights representing int4 values
        x_2d = x.reshape(-1, x.size(-1))
        w_bf16 = w.to(torch.bfloat16)
        return torch.matmul(x_2d, w_bf16)


# Default shape: LLama-2 70B attn.wqkv with B=1, L=1
B = 1
L = 1
DIN = 8192
DOUT = 1280
DTYPE = torch.bfloat16


def get_inputs():
    x = torch.randn(B, L, DIN, dtype=DTYPE)
    w = torch.randint(-8, 7, (DIN, DOUT), dtype=torch.int32)
    return [x, w]


def get_init_inputs():
    return []


def get_test_inputs():
    x = torch.randn(B, L, DIN, dtype=DTYPE, device="cuda")
    w = torch.randint(-8, 7, (DIN, DOUT), dtype=torch.int32, device="cuda")
    return [x, w]


def run(*args):
    if args:
        inputs = list(args)
    else:
        inputs = get_test_inputs()
    model = Model(*get_init_inputs()).cuda().eval()
    with torch.no_grad():
        return model(*inputs)
