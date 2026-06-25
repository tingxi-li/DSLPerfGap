import os
import sys

# Make sure the bundled CUDA libs (from the venv's nvidia-* wheels) are findable.
def _setup_cuda_libs():
    import glob
    import ctypes
    base = os.path.join(sys.prefix, "lib", "python3.10", "site-packages", "nvidia")
    if not os.path.isdir(base):
        return
    dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(base, "*", "lib", "*.so*"))})
    # Preload the core runtime libs so tilelang/tvm can dlopen against them.
    for name in ("libcudart.so.12", "libnvrtc.so.12"):
        for d in dirs:
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                try:
                    ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


_setup_cuda_libs()

import time
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[1])
def _max_kernel(M, N, block_M=16, bN=512, threads=128):
    @T.prim_func
    def func(
        A: T.Tensor((M, N), "float16"),
        Out: T.Tensor((M,), "float16"),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            chunk = T.alloc_fragment((block_M, bN), "float16")
            chunk_max = T.alloc_fragment((block_M,), "float32")
            run_max = T.alloc_fragment((block_M,), "float32")
            out_f = T.alloc_fragment((block_M,), "float16")

            for i in T.Parallel(block_M):
                run_max[i] = T.float32(-3.0e38)

            for k in T.Pipelined(T.ceildiv(N, bN), num_stages=3):
                T.copy(A[bx * block_M, k * bN], chunk)
                T.reduce_max(chunk, chunk_max, dim=1)
                for i in T.Parallel(block_M):
                    run_max[i] = T.max(run_max[i], chunk_max[i])

            for i in T.Parallel(block_M):
                out_f[i] = T.cast(run_max[i], "float16")
            T.copy(out_f, Out[bx * block_M])

    return func


_KERNEL_CACHE = {}


def _get_kernel(M, N, block_M, bN, threads):
    key = (M, N, block_M, bN, threads)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _max_kernel(M, N, block_M, bN, threads)
    return _KERNEL_CACHE[key]


def run(x, block_M=16, bN=512, threads=128):
    """Max over the last dim. Mirrors torch.max(x, dim=1).values semantics."""
    assert x.dim() == 2, "this optimized kernel targets the 2D (M, N) benchmark shape"
    x = x.contiguous()
    if x.dtype != torch.float16:
        x = x.half()
    M, N = x.shape

    # Pad N up to a multiple of bN with -inf so partial tiles are harmless.
    N_pad = ((N + bN - 1) // bN) * bN
    if N_pad != N:
        xp = torch.full((M, N_pad), float("-inf"), device=x.device, dtype=torch.float16)
        xp[:, :N] = x
        x = xp

    M_pad = ((M + block_M - 1) // block_M) * block_M
    if M_pad != M:
        xp = torch.full((M_pad, x.shape[1]), float("-inf"), device=x.device, dtype=torch.float16)
        xp[:M, :] = x
        x = xp

    kernel = _get_kernel(x.shape[0], x.shape[1], block_M, bN, threads)
    out = kernel(x)
    return out[:M]


def get_inputs():
    return [torch.randn(8192, 32768, dtype=torch.float16, device="cuda")]


def _bench(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--block_M", type=int, default=16)
    p.add_argument("--bN", type=int, default=512)
    p.add_argument("--threads", type=int, default=128)
    args = p.parse_args()

    (x,) = get_inputs()

    ref = torch.max(x, dim=1).values
    out = run(x, block_M=args.block_M, bN=args.bN, threads=args.threads)

    max_abs_err = (out.float() - ref.float()).abs().max().item()
    correct = torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    kernel_ms = _bench(lambda: run(x, block_M=args.block_M, bN=args.bN, threads=args.threads))
    pytorch_ms = _bench(lambda: torch.max(x, dim=1).values)

    elib = pytorch_ms / kernel_ms * 100.0
    print(
        f"RESULT ELIB={elib:.2f} ERR={max_abs_err:.6f} "
        f"KERNEL_MS={kernel_ms:.4f} PYTORCH_MS={pytorch_ms:.4f} CORRECT={correct}"
    )
