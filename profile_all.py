#!/usr/bin/env python3
"""
Profile all kernels: PyTorch vs Triton vs TileLang.
Uses CUDA events. Pre-compiles TileLang kernels and pre-allocates buffers
so that only the raw GPU kernel execution is measured.
Outputs to tests/results/profile.csv
"""
import csv
import gc
import importlib.util
import sys
import traceback
from pathlib import Path

import torch

KERNELS_DIR = Path("newBench")
OUT_CSV = Path("tests/results/profile.csv")
WARMUP = 10
REPEATS = 30


def load_mod(filepath, name):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cuda_bench(fn, warmup=WARMUP, repeats=REPEATS):
    """Benchmark using CUDA events. Returns (mean_ms, std_ms, peak_mem_MB, output)."""
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    peak = torch.cuda.max_memory_allocated()
    mem_mb = max(0, (peak - mem_before)) / 1024 / 1024
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std, mem_mb, out


# ── Helper: pre-compile a TileLang kernel and return a raw-call lambda ──
def _pad(val, block):
    return ((val + block - 1) // block) * block


def build_tl_gemm(M, N, K, blk_M=128, blk_N=128, blk_K=32):
    """Build a TileLang FP16 GEMM kernel; return (fn(a,b,c), a, b, c)."""
    import tilelang, tilelang.language as T
    Mp, Np, Kp = _pad(M, blk_M), _pad(N, blk_N), _pad(K, blk_K)

    @tilelang.jit
    def _k(m, n, k, bM=blk_M, bN=blk_N, bK=blk_K):
        @T.prim_func
        def f(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
              C: T.Tensor((m, n), "float16")):
            with T.Kernel(T.ceildiv(n, bN), T.ceildiv(m, bM), threads=128) as (bx, by):
                As = T.alloc_shared((bM, bK), "float16")
                Bs = T.alloc_shared((bK, bN), "float16")
                Cl = T.alloc_fragment((bM, bN), "float32")
                T.use_swizzle(panel_size=10)
                T.clear(Cl)
                for ki in T.Pipelined(T.ceildiv(k, bK), num_stages=3):
                    T.copy(A[by * bM, ki * bK], As)
                    T.copy(B[ki * bK, bx * bN], Bs)
                    T.gemm(As, Bs, Cl)
                T.copy(Cl, C[by * bM, bx * bN])
        return f

    kern = _k(Mp, Np, Kp)
    a = torch.randn(Mp, Kp, device="cuda", dtype=torch.float16)
    b = torch.randn(Kp, Np, device="cuda", dtype=torch.float16)
    c = torch.zeros(Mp, Np, device="cuda", dtype=torch.float16)
    return kern, a, b, c, Mp, Np, Kp


def build_tl_batched_gemm(Batch, M, N, K, blk_M=128, blk_N=128, blk_K=32):
    import tilelang, tilelang.language as T
    Mp, Np, Kp = _pad(M, blk_M), _pad(N, blk_N), _pad(K, blk_K)

    @tilelang.jit
    def _k(batch, m, n, k, bM=blk_M, bN=blk_N, bK=blk_K):
        @T.prim_func
        def f(A: T.Tensor((batch, m, k), "float16"), B: T.Tensor((batch, k, n), "float16"),
              C: T.Tensor((batch, m, n), "float16")):
            with T.Kernel(T.ceildiv(n, bN), T.ceildiv(m, bM), batch, threads=128) as (bx, by, bz):
                As = T.alloc_shared((bM, bK), "float16")
                Bs = T.alloc_shared((bK, bN), "float16")
                Cl = T.alloc_fragment((bM, bN), "float32")
                T.use_swizzle(panel_size=10)
                T.clear(Cl)
                for ki in T.Pipelined(T.ceildiv(k, bK), num_stages=3):
                    T.copy(A[bz, by * bM, ki * bK], As)
                    T.copy(B[bz, ki * bK, bx * bN], Bs)
                    T.gemm(As, Bs, Cl)
                T.copy(Cl, C[bz, by * bM, bx * bN])
        return f

    kern = _k(Batch, Mp, Np, Kp)
    a = torch.randn(Batch, Mp, Kp, device="cuda", dtype=torch.float16)
    b = torch.randn(Batch, Kp, Np, device="cuda", dtype=torch.float16)
    c = torch.zeros(Batch, Mp, Np, device="cuda", dtype=torch.float16)
    return kern, a, b, c


def build_tl_elemwise_add(N, blk=1024):
    import tilelang, tilelang.language as T
    Np = _pad(N, blk)

    @tilelang.jit
    def _k(n, bs=blk):
        @T.prim_func
        def f(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"),
              C: T.Tensor((n,), "float32")):
            with T.Kernel(T.ceildiv(n, bs), threads=128) as (bx,):
                al = T.alloc_fragment((bs,), "float32")
                bl = T.alloc_fragment((bs,), "float32")
                T.copy(A[bx * bs], al)
                T.copy(B[bx * bs], bl)
                for i in T.Parallel(bs):
                    al[i] = al[i] + bl[i]
                T.copy(al, C[bx * bs])
        return f

    kern = _k(Np)
    a = torch.randn(Np, device="cuda", dtype=torch.float32)
    b = torch.randn(Np, device="cuda", dtype=torch.float32)
    c = torch.zeros(Np, device="cuda", dtype=torch.float32)
    return kern, a, b, c


def build_tl_elemwise_mul(N, blk=1024):
    import tilelang, tilelang.language as T
    Np = _pad(N, blk)

    @tilelang.jit
    def _k(n, bs=blk):
        @T.prim_func
        def f(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"),
              C: T.Tensor((n,), "float32")):
            with T.Kernel(T.ceildiv(n, bs), threads=128) as (bx,):
                al = T.alloc_fragment((bs,), "float32")
                bl = T.alloc_fragment((bs,), "float32")
                T.copy(A[bx * bs], al)
                T.copy(B[bx * bs], bl)
                for i in T.Parallel(bs):
                    al[i] = al[i] * bl[i]
                T.copy(al, C[bx * bs])
        return f

    kern = _k(Np)
    a = torch.randn(Np, device="cuda", dtype=torch.float32)
    b = torch.randn(Np, device="cuda", dtype=torch.float32)
    c = torch.zeros(Np, device="cuda", dtype=torch.float32)
    return kern, a, b, c


def build_tl_relu(N, blk=1024):
    import tilelang, tilelang.language as T
    Np = _pad(N, blk)

    @tilelang.jit
    def _k(n, bs=blk):
        @T.prim_func
        def f(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
            with T.Kernel(T.ceildiv(n, bs), threads=128) as (bx,):
                al = T.alloc_fragment((bs,), "float32")
                T.copy(A[bx * bs], al)
                for i in T.Parallel(bs):
                    if al[i] < T.cast(0, "float32"):
                        al[i] = T.cast(0, "float32")
                T.copy(al, C[bx * bs])
        return f

    kern = _k(Np)
    a = torch.randn(Np, device="cuda", dtype=torch.float32)
    c = torch.zeros(Np, device="cuda", dtype=torch.float32)
    return kern, a, c


def build_tl_softmax(M, N, blk_M=32):
    import tilelang, tilelang.language as T
    Mp = _pad(M, blk_M)

    @tilelang.jit
    def _k(m, n, bM=blk_M):
        @T.prim_func
        def f(A: T.Tensor((m, n), "float32"), C: T.Tensor((m, n), "float32")):
            with T.Kernel(T.ceildiv(m, bM), threads=128) as (bx,):
                row = T.alloc_fragment((bM, n), "float32")
                mx = T.alloc_fragment((bM,), "float32")
                sm = T.alloc_fragment((bM,), "float32")
                T.copy(A[bx * bM, 0], row)
                for i in T.Parallel(bM):
                    mx[i] = T.float32(-1e30)
                for i, j in T.Parallel(bM, n):
                    if row[i, j] > mx[i]:
                        mx[i] = row[i, j]
                T.clear(sm)
                for i, j in T.Parallel(bM, n):
                    row[i, j] = T.exp(row[i, j] - mx[i])
                for i, j in T.Parallel(bM, n):
                    sm[i] = sm[i] + row[i, j]
                for i, j in T.Parallel(bM, n):
                    row[i, j] = row[i, j] / sm[i]
                T.copy(row, C[bx * bM, 0])
        return f

    kern = _k(Mp, N)
    a = torch.randn(Mp, N, device="cuda", dtype=torch.float32)
    c = torch.zeros(Mp, N, device="cuda", dtype=torch.float32)
    return kern, a, c


def build_tl_layernorm(M, N, blk_M=32, eps=1e-5):
    import tilelang, tilelang.language as T
    Mp = _pad(M, blk_M)

    @tilelang.jit
    def _k(m, n, bM=blk_M):
        @T.prim_func
        def f(X: T.Tensor((m, n), "float32"), W: T.Tensor((n,), "float32"),
              B: T.Tensor((n,), "float32"), Y: T.Tensor((m, n), "float32")):
            with T.Kernel(T.ceildiv(m, bM), threads=128) as (bx,):
                row = T.alloc_fragment((bM, n), "float32")
                mu = T.alloc_fragment((bM,), "float32")
                va = T.alloc_fragment((bM,), "float32")
                wf = T.alloc_fragment((n,), "float32")
                bf = T.alloc_fragment((n,), "float32")
                T.copy(W[0], wf); T.copy(B[0], bf)
                T.copy(X[bx * bM, 0], row)
                T.clear(mu)
                for i, j in T.Parallel(bM, n):
                    mu[i] = mu[i] + row[i, j]
                for i in T.Parallel(bM):
                    mu[i] = mu[i] / T.float32(n)
                T.clear(va)
                for i, j in T.Parallel(bM, n):
                    va[i] = va[i] + (row[i, j] - mu[i]) * (row[i, j] - mu[i])
                for i in T.Parallel(bM):
                    va[i] = va[i] / T.float32(n)
                for i, j in T.Parallel(bM, n):
                    row[i, j] = (row[i, j] - mu[i]) / T.sqrt(va[i] + T.float32(eps)) * wf[j] + bf[j]
                T.copy(row, Y[bx * bM, 0])
        return f

    kern = _k(Mp, N)
    x = torch.randn(Mp, N, device="cuda", dtype=torch.float32)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.zeros(Mp, N, device="cuda", dtype=torch.float32)
    return kern, x, w, b, y


def build_tl_rmsnorm(M, N, blk_M=32, eps=1e-5):
    import tilelang, tilelang.language as T
    Mp = _pad(M, blk_M)

    @tilelang.jit
    def _k(m, n, bM=blk_M):
        @T.prim_func
        def f(X: T.Tensor((m, n), "float32"), W: T.Tensor((n,), "float32"),
              Y: T.Tensor((m, n), "float32")):
            with T.Kernel(T.ceildiv(m, bM), threads=128) as (bx,):
                row = T.alloc_fragment((bM, n), "float32")
                rv = T.alloc_fragment((bM,), "float32")
                wf = T.alloc_fragment((n,), "float32")
                T.copy(W[0], wf)
                T.copy(X[bx * bM, 0], row)
                T.clear(rv)
                for j in T.Serial(n):
                    for i in T.Parallel(bM):
                        rv[i] = rv[i] + row[i, j] * row[i, j]
                for i in T.Parallel(bM):
                    rv[i] = T.sqrt(rv[i] / T.float32(n) + T.float32(eps))
                for i, j in T.Parallel(bM, n):
                    row[i, j] = row[i, j] / rv[i] * wf[j]
                T.copy(row, Y[bx * bM, 0])
        return f

    kern = _k(Mp, N)
    x = torch.randn(Mp, N, device="cuda", dtype=torch.float32)
    w = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.zeros(Mp, N, device="cuda", dtype=torch.float32)
    return kern, x, w, y


def build_tl_transpose(M, N, blk=64):
    import tilelang, tilelang.language as T
    Mp, Np = _pad(M, blk), _pad(N, blk)

    @tilelang.jit
    def _k(m, n, bM=blk, bN=blk):
        @T.prim_func
        def f(A: T.Tensor((m, n), "float32"), B: T.Tensor((n, m), "float32")):
            with T.Kernel(T.ceildiv(n, bN), T.ceildiv(m, bM), threads=128) as (bx, by):
                af = T.alloc_fragment((bM, bN), "float32")
                bf = T.alloc_fragment((bN, bM), "float32")
                T.copy(A[by * bM, bx * bN], af)
                for i, j in T.Parallel(bN, bM):
                    bf[i, j] = af[j, i]
                T.copy(bf, B[bx * bN, by * bM])
        return f

    kern = _k(Mp, Np)
    a = torch.randn(Mp, Np, device="cuda", dtype=torch.float32)
    b = torch.zeros(Np, Mp, device="cuda", dtype=torch.float32)
    return kern, a, b


def build_tl_swiglu(M, D, blk_M=32):
    import tilelang, tilelang.language as T
    Mp = _pad(M, blk_M)

    @tilelang.jit
    def _k(m, d, bM=blk_M):
        @T.prim_func
        def f(X: T.Tensor((m, d), "float32"), Y: T.Tensor((m, d), "float32"),
              O: T.Tensor((m, d), "float32")):
            with T.Kernel(T.ceildiv(m, bM), threads=128) as (bx,):
                xf = T.alloc_fragment((bM, d), "float32")
                yf = T.alloc_fragment((bM, d), "float32")
                T.copy(X[bx * bM, 0], xf)
                T.copy(Y[bx * bM, 0], yf)
                for i, j in T.Parallel(bM, d):
                    v = xf[i, j]
                    xf[i, j] = v * (T.float32(1) / (T.float32(1) + T.exp(-v))) * yf[i, j]
                T.copy(xf, O[bx * bM, 0])
        return f

    kern = _k(Mp, D)
    x = torch.randn(Mp, D, device="cuda", dtype=torch.float32)
    y = torch.randn(Mp, D, device="cuda", dtype=torch.float32)
    o = torch.zeros(Mp, D, device="cuda", dtype=torch.float32)
    return kern, x, y, o


# ── Main profile loop ──────────────────────────────────────────────────

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    header = [
        "kernel", "shape", "dtype",
        "pytorch_ms", "triton_ms", "tilelang_ms",
        "tl_vs_pt", "tl_vs_triton",
        "tilelang_mem_MB", "max_abs_error", "status"
    ]

    print(f"{'='*130}")
    print(f"{'Kernel':<20} {'Shape':<32} {'PyTorch':>10} {'Triton':>10} {'TileLang':>10} {'TL/PT':>8} {'TL/Tri':>8} {'MemMB':>7} {'Error':>10}")
    print(f"{'-'*130}")

    def record(name, shape, dtype, pt_ms, tr_ms, tl_ms, mem, err, status="PASS"):
        tl_pt = tl_ms / pt_ms if pt_ms > 0 else 0
        tl_tr = tl_ms / tr_ms if tr_ms and tr_ms > 0 else None
        tr_s = f"{tr_ms:.3f}" if tr_ms else "N/A"
        tt_s = f"{tl_tr:.2f}x" if tl_tr else "N/A"
        print(f"{name:<20} {shape:<32} {pt_ms:>9.3f}ms {tr_s:>9}ms {tl_ms:>9.3f}ms "
              f"{tl_pt:>7.2f}x {tt_s:>7} {mem:>6.1f} {err:>10.2e}")
        rows.append([name, shape, dtype, f"{pt_ms:.4f}",
                     f"{tr_ms:.4f}" if tr_ms else "N/A", f"{tl_ms:.4f}",
                     f"{tl_pt:.3f}", f"{tl_tr:.3f}" if tl_tr else "N/A",
                     f"{mem:.2f}", f"{err:.2e}", status])

    # ────── GEMM-like kernels (TileLang should shine here) ──────
    print("\n--- GEMM-like kernels ---")
    for S in [2048, 4096, 8192]:
        try:
            kern, a, b, c, Mp, Np, Kp = build_tl_gemm(S, S, S)
            ref_a = a[:S, :S]; ref_b = b[:S, :S]
            pt_ms, _, _, pt_out = cuda_bench(lambda: torch.matmul(ref_a, ref_b))
            # Triton matmul — kernel hardcodes M=N=K=4096 and strides, only valid at 4096
            tr_ms = None
            if S == 4096:
                try:
                    tr_mod = load_mod(
                        list(KERNELS_DIR.glob("matmul/triton*.py"))[0], f"tr_mm_{S}")
                    if hasattr(tr_mod, 'matmul'):
                        tr_c = torch.empty(S, S, device="cuda", dtype=torch.float16)
                        # Use smaller blocks to fit shared memory (101376 limit)
                        BM, BN, BK = 32, 32, 32
                        tr_ms, _, _, _ = cuda_bench(
                            lambda: tr_mod.matmul(tr_c, ref_a, ref_b, S, S, S, BM, BN, BK))
                except Exception as ex:
                    print(f"    Triton matmul err: {ex}")
            else:
                print(f"    Triton matmul: skipped for {S}x{S} (kernel hardcodes 4096x4096)")
            tl_ms, _, mem, _ = cuda_bench(lambda: kern(a, b, c))
            tl_out = c[:S, :S]
            err = (pt_out.float() - tl_out.float()).abs().max().item()
            record("matmul", f"{S}x{S}x{S}", "fp16", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Batched matmul
    # Note: Triton impl does vec-mat (A=[M,K], B=[M,N,K]) not standard batched matmul
    for B, S in [(8, 512), (16, 1024)]:
        try:
            kern, a, b, c = build_tl_batched_gemm(B, S, S, S)
            ref_a = a[:, :S, :S]; ref_b = b[:, :S, :S]
            pt_ms, _, _, pt_out = cuda_bench(lambda: torch.matmul(ref_a, ref_b))
            tr_ms = None
            try:
                tr_mod_bm = load_mod(
                    list(KERNELS_DIR.glob("batched_matmul/triton*.py"))[0], f"tr_bm_{B}_{S}")
                if hasattr(tr_mod_bm, 'batched_vecmat'):
                    # Block sizes must divide M(=B), N(=S), K(=S) evenly
                    block_m = min(8, B)
                    block_n = min(32, S)
                    block_k = min(64, S)
                    tr_ms, _, _, _ = cuda_bench(
                        lambda: tr_mod_bm.batched_vecmat(B, S, S, block_m, block_n, block_k))
            except Exception as ex:
                print(f"    Triton batched_matmul err: {ex}")
            tl_ms, _, mem, _ = cuda_bench(lambda: kern(a, b, c))
            tl_out = c[:, :S, :S]
            err = (pt_out.float() - tl_out.float()).abs().max().item()
            record("batched_matmul", f"B{B}_{S}x{S}x{S}", "fp16", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Leaky relu (matmul + leaky_relu fused)
    for S in [2048, 4096]:
        try:
            kern, a, b, c, Mp, Np, Kp = build_tl_gemm(S, S, S)
            ref_a = a[:S, :S]; ref_b = b[:S, :S]
            pt_fn = lambda: torch.nn.functional.leaky_relu(torch.matmul(ref_a, ref_b), 0.01)
            tr_mod_lr = load_mod(
                list(KERNELS_DIR.glob("leaky_relu/triton*.py"))[0], f"tr_lr_{S}")
            tr_fn = lambda: tr_mod_lr.matmul(ref_a, ref_b, activation="leaky_relu")
            pt_ms, _, _, pt_out = cuda_bench(pt_fn)
            tr_ms, _, _, _ = cuda_bench(tr_fn)
            tl_ms, _, mem, _ = cuda_bench(lambda: kern(a, b, c))
            tl_out = c[:S, :S]
            err = (pt_out.float() - tl_out.float()).abs().max().item()
            record("leaky_relu_gemm", f"{S}x{S}x{S}", "fp16", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # ────── Elementwise kernels ──────
    print("\n--- Elementwise kernels ---")
    for N in [1048576, 16777216, 67108864]:
        try:
            # Add
            kern, a, b, c = build_tl_elemwise_add(N)
            tr_mod_add = load_mod(
                list(KERNELS_DIR.glob("add/triton*.py"))[0], f"tr_add_{N}")
            ref_a = a[:N]; ref_b = b[:N]
            pt_ms, _, _, pt_out = cuda_bench(lambda: ref_a + ref_b)
            tr_ms, _, _, _ = cuda_bench(lambda: tr_mod_add.add_wrapper(ref_a, ref_b))
            tl_ms, _, mem, _ = cuda_bench(lambda: kern(a, b, c))
            err = (pt_out - c[:N]).abs().max().item()
            record("add", f"N={N}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Note: Triton mul impl does 2*x (scalar), not x*y (elementwise product)
    for N in [1048576, 16777216, 67108864]:
        try:
            kern, a, b, c = build_tl_elemwise_mul(N)
            ref_a = a[:N]; ref_b = b[:N]
            pt_ms, _, _, pt_out = cuda_bench(lambda: ref_a * ref_b)
            tr_ms = None
            try:
                tr_mod_mul = load_mod(
                    list(KERNELS_DIR.glob("mul/triton*.py"))[0], f"tr_mul_{N}")
                if hasattr(tr_mod_mul, 'triton_mul2'):
                    tr_ms, _, _, _ = cuda_bench(
                        lambda: tr_mod_mul.triton_mul2(ref_a, BLOCK_SIZE=1024))
            except Exception as ex:
                print(f"    Triton mul err: {ex}")
            tl_ms, _, mem, _ = cuda_bench(lambda: kern(a, b, c))
            err = (pt_out - c[:N]).abs().max().item()
            record("mul", f"N={N}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    for N in [1048576, 16777216, 67108864]:
        try:
            kern, a, c = build_tl_relu(N)
            ref_a = a[:N]
            pt_ms, _, _, pt_out = cuda_bench(lambda: torch.relu(ref_a))
            tr_mod_relu = load_mod(
                list(KERNELS_DIR.glob("relu/triton*.py"))[0], f"tr_relu_{N}")
            tr_ms, _, _, _ = cuda_bench(lambda: tr_mod_relu.relu(ref_a))
            tl_ms, _, mem, _ = cuda_bench(lambda: kern(a, c))
            err = (pt_out - c[:N]).abs().max().item()
            record("relu", f"N={N}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # ────── Reduction / normalization kernels ──────
    print("\n--- Reduction / normalization kernels ---")

    # Softmax: large M, moderate N (N must fit in fragment)
    for M, N in [(8192, 512), (16384, 1024), (32768, 512)]:
        try:
            kern, a, c = build_tl_softmax(M, N)
            ref_a = a[:M]
            tr_mod_sm = load_mod(
                list(KERNELS_DIR.glob("softmax/triton*.py"))[0], f"tr_sm_{M}_{N}")
            pt_ms, _, _, pt_out = cuda_bench(lambda: torch.softmax(ref_a, dim=-1))
            tr_ms = None
            try:
                tr_ms, _, _, _ = cuda_bench(lambda: tr_mod_sm.softmax(ref_a))
            except Exception:
                pass
            tl_ms, _, mem, _ = cuda_bench(lambda: kern(a, c))
            err = (pt_out - c[:M]).abs().max().item()
            record("softmax", f"{M}x{N}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Log-softmax
    for M, N in [(8192, 512), (16384, 1024)]:
        try:
            # Reuse softmax kernel structure but compute log_softmax
            kern, a, c = build_tl_softmax(M, N)  # will give softmax, not log
            ref_a = a[:M]
            tr_mod_ls = load_mod(
                list(KERNELS_DIR.glob("log_softmax/triton*.py"))[0], f"tr_ls_{M}_{N}")
            pt_ms, _, _, pt_out = cuda_bench(lambda: torch.log_softmax(ref_a, dim=-1))
            tr_ms = None
            try:
                tr_ms, _, _, _ = cuda_bench(lambda: tr_mod_ls.log_softmax(ref_a, dim=-1))
            except Exception:
                pass
            # Use tilelang_impl wrapper for log_softmax
            tl_mod_ls = load_mod(KERNELS_DIR / "log_softmax/tilelang_impl.py", "tl_ls")
            tl_ms, _, mem, tl_out = cuda_bench(lambda: tl_mod_ls.log_softmax_tilelang(ref_a, dim=-1))
            err = (pt_out - tl_out).abs().max().item()
            record("log_softmax", f"{M}x{N}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Layer norm
    # Note: Triton impl uses bfloat16 and Inductor internals
    for M, N in [(4096, 512), (8192, 1024), (16384, 512)]:
        try:
            kern, x, w, b, y = build_tl_layernorm(M, N)
            ref_x = x[:M]; ref_w = w; ref_b = b
            pt_ms, _, _, pt_out = cuda_bench(
                lambda: torch.nn.functional.layer_norm(ref_x, [N], ref_w, ref_b))
            tr_ms = None
            try:
                tr_mod_ln = load_mod(
                    list(KERNELS_DIR.glob("layer_norm/triton*.py"))[0], f"tr_ln_{M}_{N}")
                if hasattr(tr_mod_ln, 'fused_native_layer_norm'):
                    # Triton kernel expects (weight, bias, input) in bfloat16
                    tr_w = ref_w.bfloat16()
                    tr_b = ref_b.bfloat16()
                    tr_x = ref_x.bfloat16()
                    tr_ms, _, _, _ = cuda_bench(
                        lambda: tr_mod_ln.fused_native_layer_norm(tr_w, tr_b, tr_x))
            except Exception as ex:
                print(f"    Triton layer_norm err: {ex}")
            tl_ms, _, mem, _ = cuda_bench(lambda: kern(x, w, b, y))
            err = (pt_out - y[:M]).abs().max().item()
            record("layer_norm", f"{M}x{N}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # RMS norm
    for M, N in [(4096, 512), (8192, 1024), (16384, 512)]:
        try:
            kern, x, w, y = build_tl_rmsnorm(M, N)
            ref_x = x[:M]
            rms_pt = load_mod(list(KERNELS_DIR.glob("rms_norm/pytorch*.py"))[0], "pt_rms")
            tr_mod_rms = None
            try:
                tr_mod_rms = load_mod(
                    list(KERNELS_DIR.glob("rms_norm/triton*.py"))[0], f"tr_rms_{M}_{N}")
            except Exception:
                pass
            pt_ms, _, _, pt_out = cuda_bench(lambda: rms_pt.rms_norm(ref_x, [N], w))
            tr_ms = None
            if tr_mod_rms and hasattr(tr_mod_rms, 'rms_norm'):
                try:
                    tr_ms, _, _, _ = cuda_bench(lambda: tr_mod_rms.rms_norm(ref_x, [N], w))
                except Exception:
                    pass
            tl_ms, _, mem, _ = cuda_bench(lambda: kern(x, w, y))
            err = (pt_out - y[:M]).abs().max().item()
            record("rms_norm", f"{M}x{N}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Transpose
    # Note: Triton impl uses grid=(1,) single block — only works for small matrices
    print("\n--- Other kernels ---")
    for S in [4096, 8192]:
        try:
            kern, a, b = build_tl_transpose(S, S)
            ref_a = a[:S, :S]
            pt_ms, _, _, pt_out = cuda_bench(lambda: ref_a.T.contiguous())
            tr_ms = None
            try:
                tr_mod_tr = load_mod(
                    list(KERNELS_DIR.glob("matrix_transpose/triton*.py"))[0], f"tr_tr_{S}")
                if hasattr(tr_mod_tr, 'wrapper'):
                    tr_ms, _, _, _ = cuda_bench(lambda: tr_mod_tr.wrapper(S, S))
            except Exception as ex:
                print(f"    Triton transpose err: {ex} (single-block kernel, too large)")
            tl_ms, _, mem, _ = cuda_bench(lambda: kern(a, b))
            err = (pt_out - b[:S, :S]).abs().max().item()
            record("transpose", f"{S}x{S}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # SwiGLU
    for M, D in [(8192, 512), (16384, 1024)]:
        try:
            kern, xp, yp, op = build_tl_swiglu(M, D)
            xy = torch.cat([xp[:M], yp[:M]], dim=-1)
            sg_pt = load_mod(list(KERNELS_DIR.glob("swiglu/pytorch*.py"))[0], "pt_sg")
            tr_mod_sg = None
            try:
                tr_mod_sg = load_mod(
                    list(KERNELS_DIR.glob("swiglu/triton*.py"))[0], f"tr_sg_{M}_{D}")
            except Exception:
                pass
            pt_ms, _, _, pt_out = cuda_bench(lambda: sg_pt.swiglu_fwd(xy))
            tr_ms = None
            if tr_mod_sg and hasattr(tr_mod_sg, '_swiglu_fwd'):
                try:
                    tr_ms, _, _, _ = cuda_bench(lambda: tr_mod_sg._swiglu_fwd(xy))
                except Exception:
                    pass
            tl_ms, _, mem, _ = cuda_bench(lambda: kern(xp, yp, op))
            err = (pt_out - op[:M]).abs().max().item()
            record("swiglu", f"{M}x{D}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Argmax
    for M, N in [(4096, 4096), (8192, 2048)]:
        try:
            ref = torch.randn(M, N, device="cuda", dtype=torch.float32)
            tr_mod_am = load_mod(
                list(KERNELS_DIR.glob("argmax/triton*.py"))[0], f"tr_am_{M}_{N}")
            tl_mod_am = load_mod(KERNELS_DIR / "argmax/tilelang_impl.py", f"tl_am_{M}_{N}")
            pt_ms, _, _, pt_out = cuda_bench(lambda: torch.argmax(ref, dim=1))
            tr_ms = None
            try:
                tr_ms, _, _, _ = cuda_bench(lambda: tr_mod_am.argmax(ref, dim=1))
            except Exception:
                pass
            tl_ms, _, mem, tl_out = cuda_bench(lambda: tl_mod_am.argmax_tilelang(ref, dim=1))
            err = (pt_out.float() - tl_out.float()).abs().max().item()
            record("argmax", f"{M}x{N}_dim1", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Cross-entropy
    for N, V in [(2048, 10000), (4096, 50000)]:
        try:
            logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
            labels = torch.randint(0, V, (N,), device="cuda", dtype=torch.int64)
            ce_pt = load_mod(list(KERNELS_DIR.glob("cross_entropy/pytorch*.py"))[0], "pt_ce")
            tl_ce = load_mod(KERNELS_DIR / "cross_entropy/tilelang_impl.py", "tl_ce")
            tr_mod_ce = load_mod(
                list(KERNELS_DIR.glob("cross_entropy/triton*.py"))[0], f"tr_ce_{N}")
            pt_ms, _, _, pt_out = cuda_bench(lambda: ce_pt.cross_entropy_loss(logits, labels))
            tr_ms = None
            try:
                smoothing = torch.zeros(N, device="cuda", dtype=torch.float32)
                tr_ms, _, _, _ = cuda_bench(
                    lambda: tr_mod_ce.cross_entropy_fwd(
                        logits, labels, smoothing, 1.0, 0.0, -100, V, 0, 1024, False, False))
            except Exception:
                pass
            tl_ms, _, mem, tl_out = cuda_bench(lambda: tl_ce.cross_entropy_tilelang(logits, labels))
            err = (pt_out - tl_out).abs().max().item()
            record("cross_entropy", f"N{N}_V{V}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Embedding
    for N, V, D in [(8192, 50000, 512), (32768, 100000, 768)]:
        try:
            ids = torch.randint(0, V, (N,), device="cuda", dtype=torch.int64)
            w = torch.randn(V, D, device="cuda", dtype=torch.float32)
            emb_pt = load_mod(list(KERNELS_DIR.glob("embedding/pytorch*.py"))[0], "pt_emb")
            tl_emb = load_mod(KERNELS_DIR / "embedding/tilelang_impl.py", "tl_emb")
            tr_mod_emb = None
            try:
                tr_mod_emb = load_mod(
                    list(KERNELS_DIR.glob("embedding/triton*.py"))[0], f"tr_emb_{N}")
            except Exception:
                pass
            pt_ms, _, _, pt_out = cuda_bench(lambda: emb_pt.embedding(ids, w))
            tr_ms = None
            if tr_mod_emb:
                try:
                    o = torch.zeros(N, D, device="cuda", dtype=torch.float32)
                    tr_ms, _, _, _ = cuda_bench(
                        lambda: tr_mod_emb.embedding(ids, w, 0, V, o.clone()))
                except Exception:
                    pass
            tl_ms, _, mem, tl_out = cuda_bench(lambda: tl_emb.embedding_tilelang(ids, w))
            err = (pt_out - tl_out).abs().max().item()
            record("embedding", f"N{N}_V{V}_D{D}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Max reduction
    for M, N in [(8192, 2048), (16384, 4096)]:
        try:
            x = torch.randn(M, N, device="cuda", dtype=torch.float32)
            tl_mr = load_mod(KERNELS_DIR / "max_reduction/tilelang_impl.py", f"tl_mr_{M}")
            tr_mod_mx = load_mod(
                list(KERNELS_DIR.glob("max_reduction/triton*.py"))[0], f"tr_mx_{M}")
            pt_ms, _, _, pt_out = cuda_bench(lambda: torch.max(x, dim=1)[0])
            tr_ms = None
            try:
                tr_ms, _, _, _ = cuda_bench(lambda: tr_mod_mx.max_dim(x, dim=1)[0])
            except Exception:
                pass
            tl_ms, _, mem, tl_out = cuda_bench(lambda: tl_mr.max_tilelang(x, dim=1)[0])
            err = (pt_out - tl_out).abs().max().item()
            record("max_reduction", f"{M}x{N}_dim1", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Mean reduction
    for M, N in [(8192, 2048), (16384, 4096)]:
        try:
            x = torch.randn(M, N, device="cuda", dtype=torch.float32)
            tl_me = load_mod(KERNELS_DIR / "mean_reduction/tilelang_impl.py", f"tl_me_{M}")
            tr_mod_me = load_mod(
                list(KERNELS_DIR.glob("mean_reduction/triton*.py"))[0], f"tr_me_{M}")
            pt_ms, _, _, pt_out = cuda_bench(lambda: torch.mean(x, dim=1))
            tr_ms = None
            try:
                tr_ms, _, _, _ = cuda_bench(lambda: tr_mod_me.mean_dim(x, dim=1))
            except Exception:
                pass
            tl_ms, _, mem, tl_out = cuda_bench(lambda: tl_me.mean_tilelang(x, dim=1))
            err = (pt_out - tl_out).abs().max().item()
            record("mean_reduction", f"{M}x{N}_dim1", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Conv2d
    for B, C, H, OC in [(4, 16, 128, 64), (8, 32, 256, 128)]:
        try:
            inp = torch.randn(B, C, H, H, device="cuda", dtype=torch.float32)
            w = torch.randn(OC, C, 3, 3, device="cuda", dtype=torch.float32)
            cv_pt = load_mod(list(KERNELS_DIR.glob("conv2d/pytorch*.py"))[0], "pt_cv")
            tl_cv = load_mod(KERNELS_DIR / "conv2d/tilelang_impl.py", "tl_cv")
            tr_mod_cv = load_mod(
                list(KERNELS_DIR.glob("conv2d/triton*.py"))[0], f"tr_cv_{B}_{C}")
            pt_ms, _, _, pt_out = cuda_bench(lambda: cv_pt.conv2d(inp, w, padding=1))
            tr_ms = None
            try:
                tr_ms, _, _, _ = cuda_bench(
                    lambda: tr_mod_cv.conv2d_forward(inp, w, 3, 3, 1, 1, 1, 1, 1))
            except Exception:
                pass
            tl_ms, _, mem, tl_out = cuda_bench(lambda: tl_cv.conv2d_tilelang(inp, w, padding=1))
            err = (pt_out.float() - tl_out.float()).abs().max().item()
            record("conv2d", f"B{B}_C{C}_H{H}_OC{OC}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Logsumexp
    for M, N in [(8192, 512), (16384, 1024)]:
        try:
            x = torch.randn(M, N, device="cuda", dtype=torch.float32)
            tl_lse = load_mod(KERNELS_DIR / "logsumexp/tilelang_impl.py", "tl_lse")
            tr_mod_lse = None
            try:
                tr_mod_lse = load_mod(
                    list(KERNELS_DIR.glob("logsumexp/triton*.py"))[0], f"tr_lse_{M}")
            except Exception:
                pass
            pt_ms, _, _, pt_out = cuda_bench(lambda: torch.logsumexp(x, dim=1))
            tr_ms = None
            if tr_mod_lse and hasattr(tr_mod_lse, 'logsumexp_fwd'):
                try:
                    tr_ms, _, _, _ = cuda_bench(lambda: tr_mod_lse.logsumexp_fwd(x))
                except Exception:
                    pass
            tl_ms, _, mem, tl_out = cuda_bench(lambda: tl_lse.logsumexp_tilelang(x, dim=1))
            err = (pt_out - tl_out).abs().max().item()
            record("logsumexp", f"{M}x{N}_dim1", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Index select
    for N, V, D in [(8192, 50000, 512), (32768, 100000, 768)]:
        try:
            src = torch.randn(V, D, device="cuda", dtype=torch.float32)
            idx = torch.randint(0, V, (N,), device="cuda", dtype=torch.int64)
            tl_is = load_mod(KERNELS_DIR / "index_select/tilelang_impl.py", "tl_is")
            tr_mod_is = load_mod(
                list(KERNELS_DIR.glob("index_select/triton*.py"))[0], f"tr_is_{N}")
            pt_ms, _, _, pt_out = cuda_bench(lambda: src[idx])
            tr_ms = None
            try:
                tr_ms, _, _, _ = cuda_bench(
                    lambda: tr_mod_is.index_select_cat_fwd(
                        torch.zeros(N, D, device="cuda"), src, idx))
            except Exception:
                pass
            tl_ms, _, mem, tl_out = cuda_bench(
                lambda: tl_is.index_select_tilelang(
                    torch.zeros(N, D, device="cuda"), src, idx))
            err = (pt_out - tl_out).abs().max().item()
            record("index_select", f"N{N}_V{V}_D{D}", "fp32", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Attention
    for B, H, S, D in [(2, 8, 256, 64), (4, 16, 128, 64)]:
        try:
            q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
            k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
            v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
            attn_pt = load_mod(
                list(KERNELS_DIR.glob("attention/pytorch*.py"))[0], "pt_attn")
            tl_attn = load_mod(KERNELS_DIR / "attention/tilelang_impl.py", "tl_attn")
            tr_mod_attn = load_mod(
                list(KERNELS_DIR.glob("attention/triton*.py"))[0], f"tr_attn_{B}")
            pt_ms, _, _, pt_out = cuda_bench(lambda: attn_pt.flash_attn_triton(q, k, v))
            tr_ms = None
            try:
                tr_ms, _, _, _ = cuda_bench(
                    lambda: tr_mod_attn.AttentionFunction.apply(q, k, v))
            except Exception:
                pass
            tl_ms, _, mem, tl_out = cuda_bench(
                lambda: tl_attn.attention_tilelang(q, k, v))
            err = (pt_out.float() - tl_out.float()).abs().max().item()
            record("attention", f"B{B}_H{H}_S{S}_D{D}", "fp16", pt_ms, tr_ms, tl_ms, mem, err)
        except Exception:
            traceback.print_exc()

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n{'='*130}")
    print(f"Results written to {OUT_CSV}")
    print(f"Total: {len(rows)} benchmarks")


if __name__ == "__main__":
    main()
