import torch

def batched_matmul(A, B):
    """
    Batched vector-matrix product.
    A: [M, K], B: [M, N, K]
    Output[m, n] = sum_k(A[m, k] * B[m, n, k])
    """
    return torch.einsum('mk,mnk->mn', A, B)
