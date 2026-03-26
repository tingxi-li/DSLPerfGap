import torch
import torch.nn as nn

# --- Original implementation inlined below ---
"""
TileLang conv2d via im2col + GEMM.
Steps:
  1. Use torch.nn.functional.unfold (im2col) to extract patches -> (N, C*KH*KW, OH*OW)
  2. Reshape weight to (OC, C*KH*KW)
  3. Perform GEMM using TileLang: weight @ col -> (N, OC, OH*OW)
  4. Reshape to (N, OC, OH, OW) and add bias if present
"""
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
NUM_STAGES = 2
NUM_THREADS = 128


def _tilelang_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A: (M, K), B: (K, N) -> C: (M, N).
    Uses float16 inputs with float32 accumulation to leverage tensor cores.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    block_M = BLOCK_M
    block_N = BLOCK_N
    block_K = BLOCK_K

    # Pad to multiples of block sizes
    M_pad = ((M + block_M - 1) // block_M) * block_M
    N_pad = ((N + block_N - 1) // block_N) * block_N
    K_pad = ((K + block_K - 1) // block_K) * block_K

    A_f16 = A.half()
    B_f16 = B.half()

    if M_pad != M or K_pad != K:
        A_pad = torch.zeros(M_pad, K_pad, device=A.device, dtype=torch.float16)
        A_pad[:M, :K] = A_f16
    else:
        A_pad = A_f16.contiguous()

    if K_pad != K or N_pad != N:
        B_pad = torch.zeros(K_pad, N_pad, device=B.device, dtype=torch.float16)
        B_pad[:K, :N] = B_f16
    else:
        B_pad = B_f16.contiguous()

    C_pad = torch.zeros(M_pad, N_pad, device=A.device, dtype=torch.float32)

    @tilelang.jit
    def kernel(M_dim, N_dim, K_dim, bM=block_M, bN=block_N, bK=block_K):
        @T.prim_func
        def func(
            A_t: T.Tensor((M_dim, K_dim), "float16"),
            B_t: T.Tensor((K_dim, N_dim), "float16"),
            C_t: T.Tensor((M_dim, N_dim), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(N_dim, bN), T.ceildiv(M_dim, bM), threads=NUM_THREADS
            ) as (bx, by):
                A_shared = T.alloc_shared((bM, bK), "float16")
                B_shared = T.alloc_shared((bK, bN), "float16")
                C_local = T.alloc_fragment((bM, bN), "float32")

                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K_dim, bK), num_stages=NUM_STAGES):
                    T.copy(A_t[by * bM, k * bK], A_shared)
                    T.copy(B_t[k * bK, bx * bN], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                T.copy(C_local, C_t[by * bM, bx * bN])

        return func

    fn = kernel(M_pad, N_pad, K_pad)
    fn(A_pad, B_pad, C_pad)

    return C_pad[:M, :N]


def conv2d(input, weight, bias=None, stride=1, padding=0, groups=1):
    """
    Conv2d using im2col (torch.nn.functional.unfold) + TileLang GEMM.
    Note: dilation is not supported (always 1).
    input:  (N, C, H, W)
    weight: (OC, C/groups, KH, KW)
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    dilation = (1, 1)

    N_batch, C_in, H, W = input.shape
    OC, C_per_group, KH, KW = weight.shape

    # Output spatial dimensions
    OH = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
    OW = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1

    # Use unfold (im2col): produces (N, C*KH*KW, OH*OW)
    col = F.unfold(input, kernel_size=(KH, KW), dilation=dilation,
                   padding=padding, stride=stride)

    # Reshape weight: (OC, C_per_group*KH*KW)
    weight_mat = weight.reshape(OC, C_per_group * KH * KW).contiguous()

    # Ensure float32 for GEMM
    col = col.float()
    weight_mat = weight_mat.float()

    if groups == 1:
        results = []
        for n in range(N_batch):
            col_n = col[n].contiguous()
            out_n = _tilelang_gemm(weight_mat, col_n)
            results.append(out_n)
        output = torch.stack(results, dim=0)
    else:
        col_per_group = C_per_group * KH * KW
        oc_per_group = OC // groups
        results = []
        for n in range(N_batch):
            group_results = []
            for g in range(groups):
                col_g = col[n, g * col_per_group:(g + 1) * col_per_group, :].contiguous()
                w_g = weight_mat[g * oc_per_group:(g + 1) * oc_per_group, :].contiguous()
                out_g = _tilelang_gemm(w_g, col_g)
                group_results.append(out_g)
            out_n = torch.cat(group_results, dim=0)
            results.append(out_n)
        output = torch.stack(results, dim=0)

    output = output.reshape(N_batch, OC, OH, OW)

    if bias is not None:
        output = output + bias.reshape(1, OC, 1, 1)

    return output

# --- End original implementation ---


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return conv2d(*args)


def get_inputs():
    x = torch.randn(32, 256, 128, 128, device='cuda', dtype=torch.float16)
    w = torch.randn(256, 256, 3, 3, device='cuda', dtype=torch.float16)
    return [x, w]

def get_init_inputs():
    return []
