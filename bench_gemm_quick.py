import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def _gemm(M, N, K, block_M=128, block_N=128, block_K=32):
    @T.prim_func
    def func(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return func


def bench(M, N, K, reps=20):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    kernel = _gemm(M, N, K)

    # Warmup
    for _ in range(5):
        kernel(a, b, c)
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # TileLang
    times = []
    for _ in range(reps):
        start.record()
        kernel(a, b, c)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    tl_ms = sum(times) / len(times)

    # PyTorch
    for _ in range(5):
        torch.matmul(a, b)
        torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start.record()
        torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    pt_ms = sum(times) / len(times)

    print(f"  M=N=K={M:5d}  PyTorch={pt_ms:.3f}ms  TileLang={tl_ms:.3f}ms  speedup={pt_ms/tl_ms:.2f}x")


if __name__ == "__main__":
    for size in [1024, 2048, 4096, 8192]:
        bench(size, size, size)
