import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(input)


# Default shapes from operator.py: first iter is (B=32, T=512, D=768, V=1024)
B = 32
T = 512
D = 768
V = 1024


def get_inputs():
    return [torch.randint(0, V, (B, T))]


def get_init_inputs():
    return [V, D]


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
