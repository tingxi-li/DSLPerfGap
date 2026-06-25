import torch
import tilelang
import tilelang.language as T

# Tiled streaming (online) log-softmax over the last dim, modeled on the working
# softmax_opt kernel: vectorized T.copy bulk loads (NOT manual T.Parallel element
# loads, which were ~28x slower), fp32 accumulation, native-fp16 I/O. Pass 1
# computes a running max and a max-rescaled running sum of exp(x-max); pass 2
# writes x - (max + log(sum)).


@tilelang.jit(out_idx=-1)
def _log_softmax_kernel(M, N, block_M=1, block_N=4096, threads=256):
    dtype = "float16"
    accum = "float32"
    NK = T.ceildiv(N, block_N)

    @T.prim_func
    def func(A: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            chunk = T.alloc_fragment((block_M, block_N), accum)
            chunk_max = T.alloc_fragment((block_M,), accum)
            chunk_sum = T.alloc_fragment((block_M,), accum)
            run_max = T.alloc_fragment((block_M,), accum)
            new_max = T.alloc_fragment((block_M,), accum)
            run_sum = T.alloc_fragment((block_M,), accum)
            lse = T.alloc_fragment((block_M,), accum)

            T.fill(run_max, -T.infinity(accum))
            T.clear(run_sum)

            # Pass 1: streaming max + rescaled sum of exp(x - max)
            for k in T.serial(NK):
                T.copy(A[bx * block_M, k * block_N], chunk)
                T.reduce_max(chunk, chunk_max, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    new_max[i] = T.max(run_max[i], chunk_max[i])
                for i, j in T.Parallel(block_M, block_N):
                    chunk[i, j] = T.exp(chunk[i, j] - new_max[i])
                T.reduce_sum(chunk, chunk_sum, dim=1)
                for i in T.Parallel(block_M):
                    run_sum[i] = run_sum[i] * T.exp(run_max[i] - new_max[i]) + chunk_sum[i]
                    run_max[i] = new_max[i]

            # logsumexp per row
            for i in T.Parallel(block_M):
                lse[i] = run_max[i] + T.log(run_sum[i])

            # Pass 2: write x - lse
            for k in T.serial(NK):
                T.copy(A[bx * block_M, k * block_N], chunk)
                for i, j in T.Parallel(block_M, block_N):
                    C[bx * block_M + i, k * block_N + j] = T.cast(chunk[i, j] - lse[i], dtype)

    return func


_KCACHE = {}


def run(x):
    assert x.dim() == 2 and x.is_cuda
    M, N = x.shape
    x = x.contiguous()
    key = (M, N)
    if key not in _KCACHE:
        _KCACHE[key] = _log_softmax_kernel(M, N)
    return _KCACHE[key](x)


def get_inputs():
    return [torch.randn(4096, 32768, dtype=torch.float16, device='cuda')]
