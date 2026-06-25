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

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("conv2d", "tilelang") or {}
except Exception:
    _TUNED = {}

BLOCK_M = _TUNED.get("BLOCK_M", 64)
BLOCK_N = _TUNED.get("BLOCK_N", 64)
BLOCK_K = _TUNED.get("BLOCK_K", 32)
NUM_STAGES = _TUNED.get("NUM_STAGES", 2)
NUM_THREADS = _TUNED.get("NUM_THREADS", 128)


@tilelang.jit
def _conv2d_gemm_kernel(M_dim, N_dim, K_dim, bM=BLOCK_M, bN=BLOCK_N, bK=BLOCK_K):
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

    fn = _conv2d_gemm_kernel(M_pad, N_pad, K_pad)
    fn(A_pad, B_pad, C_pad)

    return C_pad[:M, :N]


_TL_DTYPE = {torch.float16: "float16", torch.float32: "float32", torch.bfloat16: "bfloat16"}


@tilelang.jit
def _conv2d_direct_kernel(N, C, Hp, Wp, OC, KH, KW, OH, OW, bM, bN, bK, threads):
    """Direct conv (stride 1) as KH*KW accumulating fp16 tensor-core GEMMs over a
    spatially padded input. One block owns an output row (bN == OW) for OC tile
    bM. For each tap (kh,kw) and channel tile, the input slice is a contiguous,
    bounds-free, coalesced affine region of the padded input -- no im2col
    materialization, no per-element index math, no fp32 round-trip on B.
    """
    KHW = KH * KW
    NROWS = N * OH

    @T.prim_func
    def func(Xp: T.Tensor((N, C, Hp, Wp), "float16"),
             Wt: T.Tensor((KHW, OC, C), "float16"),
             Ct: T.Tensor((N, OC, OH, OW), "float16")):
        with T.Kernel(NROWS, T.ceildiv(OC, bM), threads=threads) as (bx, by):
            n = bx // OH
            oh = bx % OH
            As = T.alloc_shared((bM, bK), "float16")
            Bs = T.alloc_shared((bK, bN), "float16")
            Cl = T.alloc_fragment((bM, bN), "float32")
            T.clear(Cl)
            for t in T.serial(KHW):
                kh = t // KW
                kw = t % KW
                for kc in T.serial(T.ceildiv(C, bK)):
                    T.copy(Wt[t, by * bM, kc * bK], As)
                    for ci, wj in T.Parallel(bK, bN):
                        Bs[ci, wj] = Xp[n, kc * bK + ci, oh + kh, kw + wj]
                    T.gemm(As, Bs, Cl)
            # write straight into NCHW output (no post-kernel transpose)
            for i, j in T.Parallel(bM, bN):
                Ct[n, by * bM + i, oh, j] = T.cast(Cl[i, j], "float16")
    return func


def _conv2d_direct(input, weight, bias, padding, OH, OW):
    N_, C_, Hh, Ww = input.shape
    OC, _, KH, KW = weight.shape
    out_dtype = input.dtype
    x = input if input.dtype == torch.float16 else input.to(torch.float16)
    w16 = weight if weight.dtype == torch.float16 else weight.to(torch.float16)
    xpad = F.pad(x, (padding[1], padding[1], padding[0], padding[0])).contiguous()
    Hp, Wp = Hh + 2 * padding[0], Ww + 2 * padding[1]
    wt = w16.permute(2, 3, 0, 1).reshape(KH * KW, OC, C_).contiguous()

    bM = OC
    bN = OW
    if C_ % 128 == 0:
        bK = 128
    elif C_ % 64 == 0:
        bK = 64
    elif C_ % 32 == 0:
        bK = 32
    else:
        bK = 16

    out = torch.empty(N_, OC, OH, OW, device=x.device, dtype=torch.float16)
    _conv2d_direct_kernel(N_, C_, Hp, Wp, OC, KH, KW, OH, OW, bM, bN, bK, 256)(xpad, wt, out)
    if bias is not None:
        out = out + bias.reshape(1, OC, 1, 1).to(torch.float16)
    return out.to(out_dtype)


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

    # Fast path: direct conv via accumulating fp16 tensor-core GEMMs (no im2col).
    # Restricted to tensor-core-friendly, stride-1 configs (covers the large
    # benchmark); everything else falls through to the im2col path below.
    if (groups == 1 and stride == (1, 1) and dilation == (1, 1)
            and OW % 16 == 0 and OW <= 128
            and C_in % 16 == 0 and OC % 16 == 0 and OC <= 256
            and input.dtype in _TL_DTYPE):
        return _conv2d_direct(input, weight, bias, padding, OH, OW)

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
