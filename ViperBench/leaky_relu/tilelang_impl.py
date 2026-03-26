import tilelang
import tilelang.language as T
import torch


def leaky_relu(a, b, activation=""):
    """
    Matrix multiply with optional leaky_relu.
    C = leaky_relu(A @ B) if activation == "leaky_relu", else C = A @ B
    Output is float16.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    block_M, block_N, block_K = 128, 128, 32
    M_pad = ((M + block_M - 1) // block_M) * block_M
    N_pad = ((N + block_N - 1) // block_N) * block_N
    K_pad = ((K + block_K - 1) // block_K) * block_K

    do_leaky = activation == "leaky_relu"

    @tilelang.jit
    def kernel(m, n, k, bM=block_M, bN=block_N, bK=block_K):
        @T.prim_func
        def func(
            A: T.Tensor((m, k), "float16"),
            B: T.Tensor((k, n), "float16"),
            C: T.Tensor((m, n), "float16"),
        ):
            with T.Kernel(T.ceildiv(n, bN), T.ceildiv(m, bM), threads=128) as (bx, by):
                A_shared = T.alloc_shared((bM, bK), "float16")
                B_shared = T.alloc_shared((bK, bN), "float16")
                C_local = T.alloc_fragment((bM, bN), "float32")
                T.use_swizzle(panel_size=10)
                T.clear(C_local)
                for ki in T.Pipelined(T.ceildiv(k, bK), num_stages=3):
                    T.copy(A[by * bM, ki * bK], A_shared)
                    T.copy(B[ki * bK, bx * bN], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                if do_leaky:
                    for i, j in T.Parallel(bM, bN):
                        if C_local[i, j] < T.float32(0):
                            C_local[i, j] = C_local[i, j] * T.float32(0.01)
                T.copy(C_local, C[by * bM, bx * bN])
        return func

    af = a.half().contiguous()
    bf = b.half().contiguous()

    if M_pad != M or K_pad != K:
        a_pad = torch.zeros(M_pad, K_pad, device=a.device, dtype=torch.float16)
        a_pad[:M, :K] = af
    else:
        a_pad = af

    if K_pad != K or N_pad != N:
        b_pad = torch.zeros(K_pad, N_pad, device=b.device, dtype=torch.float16)
        b_pad[:K, :N] = bf
    else:
        b_pad = bf

    c_pad = torch.zeros(M_pad, N_pad, device=a.device, dtype=torch.float16)
    func = kernel(M_pad, N_pad, K_pad)
    func(a_pad, b_pad, c_pad)
    return c_pad[:M, :N]
