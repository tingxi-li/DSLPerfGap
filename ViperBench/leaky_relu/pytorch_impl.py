import torch
import torch.nn.functional as F


def leaky_relu(a, b, activation=""):
    """
    Matrix multiply with optional leaky_relu.
    C = leaky_relu(A @ B) if activation == "leaky_relu", else C = A @ B
    Output is float16.
    """
    c = torch.matmul(a.float(), b.float())
    if activation == "leaky_relu":
        c = F.leaky_relu(c, negative_slope=0.01)
    return c.half()
