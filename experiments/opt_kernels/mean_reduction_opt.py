import tilelang
import tilelang.language as T
import torch
import time


@tilelang.jit(out_idx=[1])
def _mean_kernel(m, n, bM=4, bN=4096, threads=128):
    @T.prim_func
    def func(
        A: T.Tensor((m, n), "float32"),
        Out: T.Tensor((m,), "float32"),
    ):
        with T.Kernel(T.ceildiv(m, bM), threads=threads) as bx:
            tile = T.alloc_fragment((bM, bN), "float32")
            partial = T.alloc_fragment((bM,), "float32")
            acc = T.alloc_fragment((bM,), "float32")
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(n, bN), num_stages=3):
                T.copy(A[bx * bM, k * bN], tile)
                T.reduce_sum(tile, partial, dim=1)
                for i in T.Parallel(bM):
                    acc[i] = acc[i] + partial[i]
            for i in T.Parallel(bM):
                acc[i] = acc[i] / T.float32(n)
            T.copy(acc, Out[bx * bM])
    return func


_KERN_CACHE = {}


def run(x):
    orig_shape = x.shape
    N = orig_shape[-1]
    M = x.numel() // N
    x_2d = x.contiguous().view(M, N)
    key = (M, N, x_2d.dtype)
    if key not in _KERN_CACHE:
        _KERN_CACHE[key] = _mean_kernel(M, N)
    func = _KERN_CACHE[key]
    out = func(x_2d)
    out_shape = list(orig_shape[:-1])
    return out.view(*out_shape) if out_shape else out.squeeze()


def get_inputs():
    return [torch.randn(8192, 32768, dtype=torch.float32, device='cuda')]


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
    x = get_inputs()[0]
    ref = x.mean(dim=1)
    out = run(x)
    err = (out - ref).abs().max().item()
    correct = torch.allclose(out, ref, atol=2e-3, rtol=2e-3)

    kernel_ms = _bench(lambda: run(x))
    pytorch_ms = _bench(lambda: x.mean(dim=1))
    elib = pytorch_ms / kernel_ms * 100.0

    print(f"RESULT ELIB={elib:.2f} ERR={err:.6e} KERNEL_MS={kernel_ms:.4f} "
          f"PYTORCH_MS={pytorch_ms:.4f} CORRECT={correct}")
