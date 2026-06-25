import os
import sys

# Ensure the venv-bundled CUDA libs are on the loader path (tvm/tilelang need them).
_NV = "/home/ubuntu/dslperf-venv/lib/python3.10/site-packages/nvidia"
if _NV and not os.environ.get("_SOFTMAX_REEXEC"):
    _libs = [
        f"{_NV}/cuda_nvrtc/lib", f"{_NV}/cuda_runtime/lib", f"{_NV}/cublas/lib",
        f"{_NV}/cudnn/lib", f"{_NV}/cuda_cupti/lib",
    ]
    _cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(_libs + ([_cur] if _cur else []))
    os.environ["_SOFTMAX_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import time
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=-1)
def softmax_kernel(M, N, block_M=1, block_N=4096, threads=256):
    dtype = "float16"
    accum = "float32"
    NK = T.ceildiv(N, block_N)

    @T.prim_func
    def func(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            chunk = T.alloc_fragment((block_M, block_N), accum)
            chunk_max = T.alloc_fragment((block_M,), accum)
            chunk_sum = T.alloc_fragment((block_M,), accum)
            run_max = T.alloc_fragment((block_M,), accum)
            new_max = T.alloc_fragment((block_M,), accum)
            run_sum = T.alloc_fragment((block_M,), accum)
            scale = T.alloc_fragment((block_M,), accum)

            T.fill(run_max, -T.infinity(accum))
            T.clear(run_sum)

            # ---- Pass 1: streaming max + sum over the N dimension ----
            for k in T.serial(NK):
                T.copy(A[bx * block_M, k * block_N], chunk)
                T.reduce_max(chunk, chunk_max, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    new_max[i] = T.max(run_max[i], chunk_max[i])
                for i, j in T.Parallel(block_M, block_N):
                    chunk[i, j] = T.exp(chunk[i, j] - new_max[i])
                T.reduce_sum(chunk, chunk_sum, dim=1)
                for i in T.Parallel(block_M):
                    scale[i] = T.exp(run_max[i] - new_max[i])
                    run_sum[i] = run_sum[i] * scale[i] + chunk_sum[i]
                    run_max[i] = new_max[i]

            for i in T.Parallel(block_M):
                run_sum[i] = T.float32(1.0) / run_sum[i]

            # ---- Pass 2: normalized output ----
            for k in T.serial(NK):
                T.copy(A[bx * block_M, k * block_N], chunk)
                for i, j in T.Parallel(block_M, block_N):
                    C[bx * block_M + i, k * block_N + j] = T.cast(
                        T.exp(chunk[i, j] - run_max[i]) * run_sum[i], dtype
                    )

    return func


_KCACHE = {}


def run(x):
    assert x.dim() == 2, "expects 2D (M, N)"
    M, N = x.shape
    x = x.contiguous()
    key = (M, N)
    kern = _KCACHE.get(key)
    if kern is None:
        kern = softmax_kernel(M, N)
        _KCACHE[key] = kern
    return kern(x)


def get_inputs():
    return [torch.randn(4096, 32768, dtype=torch.float16, device='cuda')]


def _bench(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


if __name__ == "__main__":
    x = get_inputs()[0]

    ref = torch.softmax(x.float(), dim=-1).half()
    out = run(x)

    max_abs_err = (out.float() - ref.float()).abs().max().item()
    correct = torch.allclose(out, ref, atol=2e-2, rtol=2e-2)

    kernel_ms = _bench(lambda: run(x))
    pytorch_ms = _bench(lambda: torch.softmax(x.float(), dim=-1).half())

    elib = pytorch_ms / kernel_ms * 100.0

    print(f"RESULT ELIB={elib:.2f} ERR={max_abs_err:.6e} "
          f"KERNEL_MS={kernel_ms:.4f} PYTORCH_MS={pytorch_ms:.4f} CORRECT={correct}")
