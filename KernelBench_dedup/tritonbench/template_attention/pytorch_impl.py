import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, batch_size: int, num_heads: int, seq_len: int, d_head: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.d_head = d_head

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: (batch, num_heads, seq_len, d_head)
        # Template attention is a standard attention pattern; use SDPA as baseline
        return F.scaled_dot_product_attention(q, k, v)


# Default shapes from operator.py: BUILDIN_SHAPES = [(16, 16, 4096, 64)]
BATCH_SIZE = 16
NUM_HEADS = 16
SEQ_LEN = 4096
D_HEAD = 64
DTYPE = torch.float16


def get_inputs():
    q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, dtype=DTYPE)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, dtype=DTYPE)
    v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, dtype=DTYPE)
    return [q, k, v]


def get_init_inputs():
    return [BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD]


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
