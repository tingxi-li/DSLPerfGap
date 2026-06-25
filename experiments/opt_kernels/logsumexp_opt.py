import os
import sys

# Ensure CUDA runtime libs from the pip nvidia packages are loadable, then re-exec once.
if os.environ.get("_LSE_REEXEC") != "1":
    _nv = os.path.join(os.path.dirname(os.__file__), "site-packages", "nvidia")
    # Fallback: derive from this interpreter's site-packages
    import site
    paths = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvd = os.path.join(sp, "nvidia")
        if os.path.isdir(nvd):
            for sub in ("cuda_nvrtc", "cuda_runtime", "cublas", "nvjitlink"):
                p = os.path.join(nvd, sub, "lib")
                if os.path.isdir(p):
                    paths.append(p)
    if paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths + [os.environ.get("LD_LIBRARY_PATH", "")])
    os.environ["_LSE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import time
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[1])
def _logsumexp_kernel(M, N, dtype="float32", bM=2, bN=2048, threads=128, num_stages=4):
    NEG = -3.0e38

    @T.prim_func
    def func(A: T.Tensor((M, N), dtype), C: T.Tensor((M,), dtype)):
        with T.Kernel(T.ceildiv(M, bM), threads=threads) as bx:
            tile = T.alloc_fragment((bM, bN), "float32")
            cmax = T.alloc_fragment((bM,), "float32")
            csum = T.alloc_fragment((bM,), "float32")
            run_max = T.alloc_fragment((bM,), "float32")
            new_max = T.alloc_fragment((bM,), "float32")
            run_sum = T.alloc_fragment((bM,), "float32")
            scale = T.alloc_fragment((bM,), "float32")

            T.fill(run_max, NEG)
            T.clear(run_sum)

            for k in T.Pipelined(T.ceildiv(N, bN), num_stages=num_stages):
                # load tile (cast to fp32 if input is lower precision)
                for i, j in T.Parallel(bM, bN):
                    tile[i, j] = T.cast(A[bx * bM + i, k * bN + j], "float32")

                # chunk max per row
                T.reduce_max(tile, cmax, dim=1, clear=True)

                # new running max
                for i in T.Parallel(bM):
                    new_max[i] = T.max(run_max[i], cmax[i])

                # exp(x - new_max)
                for i, j in T.Parallel(bM, bN):
                    tile[i, j] = T.exp(tile[i, j] - new_max[i])

                # chunk sum per row
                T.reduce_sum(tile, csum, dim=1, clear=True)

                # rescale running sum and accumulate
                for i in T.Parallel(bM):
                    scale[i] = T.exp(run_max[i] - new_max[i])
                for i in T.Parallel(bM):
                    run_sum[i] = run_sum[i] * scale[i] + csum[i]
                for i in T.Parallel(bM):
                    run_max[i] = new_max[i]

            for i in T.Parallel(bM):
                C[bx * bM + i] = T.cast(run_max[i] + T.log(run_sum[i]), dtype)

    return func


_CACHE = {}


def run(x):
    orig_shape = x.shape
    N = orig_shape[-1]
    M = x.numel() // N
    x2d = x.contiguous().view(M, N)

    dtype = "float32" if x.dtype == torch.float32 else ("float16" if x.dtype == torch.float16 else "bfloat16")

    bM = 2
    assert M % bM == 0, f"M={M} not divisible by bM={bM}"
    key = (M, N, dtype, bM)
    if key not in _CACHE:
        _CACHE[key] = _logsumexp_kernel(M, N, dtype=dtype, bM=bM)
    func = _CACHE[key]

    out = func(x2d)  # (M,)

    out_shape = list(orig_shape[:-1])
    if len(out_shape) == 0:
        return out.squeeze()
    return out.view(*out_shape)


def get_inputs():
    return [torch.randn(8192, 16384, dtype=torch.float32, device="cuda")]


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
    ref = torch.logsumexp(x, dim=-1)
    out = run(x)
    err = (out.float() - ref.float()).abs().max().item()
    correct = torch.allclose(out.float(), ref.float(), atol=2e-3, rtol=2e-3)

    kernel_ms = _bench(lambda: run(x))
    pytorch_ms = _bench(lambda: torch.logsumexp(x, dim=-1))
    elib = pytorch_ms / kernel_ms * 100.0

    print(f"RESULT ELIB={elib:.2f} ERR={err:.6e} KERNEL_MS={kernel_ms:.4f} PYTORCH_MS={pytorch_ms:.4f} CORRECT={correct}")
