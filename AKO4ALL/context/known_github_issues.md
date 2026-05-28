## TileLang T.reduce

I have searched the Issue Tracker that this hasn't already been reported. (comment there if it has.)
Motivation
T.reduce behaves poorly when the layout requires each thread to hold multiple reduced values. See the program below.

The fragment layout is satisfactory, since 8 * sizeof (bfloat16) = 16 bytes is an excellent reading width from shared memory. However, the T.reduce code does not leverage this advantage. Instead, it gets the elements from shared memory one by one and calculates the 8 max values sequentially. As we know, tl::AllReduce leverages a butterfly reduction algorithm; there are many thread syncs in the path. This simple piece of code has 8 * log(256 / 32) = 24 thread syncs! I suggest calculating these max values in parallel, i.e., only one butterfly reduction path, with 8 values done in parallel.

One may argue that parallel reduction may cost more shared memory. However, if you see the issue #1761, the issue suggests that each warp, instead of each thread, holds one copy of all the values to reduce in shared memory. The memory saved that way is just good for parallel reduction here!


```python
import tilelang
from tilelang import language as T

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def get_sample_kernel():
    @T.prim_func
    def sample_kernel(a: T.Tensor([128, 64], T.bfloat16)):
        with T.Kernel(1, threads=256):
            a_shared = T.alloc_shared([128, 64], T.bfloat16)
            amax_fragment = T.alloc_fragment([64, ], T.bfloat16)
            T.copy(a, a_shared)
            T.reduce_max(a_shared, amax_fragment, dim=0)

    return sample_kernel

kernel = get_sample_kernel()
print(kernel.get_kernel_source())
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void sample_kernel_kernel(const bfloat16_t* __restrict__ a);
extern "C" __global__ void __launch_bounds__(256, 1) sample_kernel_kernel(const bfloat16_t* __restrict__ a) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  bfloat16_t a_shared_frag[32];
  bfloat16_t amax_fragment[8];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((i * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(a + ((i * 2048) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(uint4*)(a_shared_frag + (i_1 * 8)) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((i_1 * 2048) + (((int)threadIdx.x) * 8)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    amax_fragment[i_2] = -std::numeric_limits<bfloat16_t>::infinity();
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      amax_fragment[i_2] = cutlass::bfloat16_t(__hmax((amax_fragment[i_2]).to_nv_bfloat16(), (a_shared_frag[((rv * 8) + i_2)]).to_nv_bfloat16()));
    }
    amax_fragment[i_2] = tl::AllReduce<tl::MaxOp, 256, 8, 0, tl::NamedBarrier<256>>::run(amax_fragment[i_2], (&(((bfloat16_t*)buf_dyn_shmem)[0])));
  }
}
```

### TileLang Float32

```python
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def single_gemm_kernel(
    M, N, K,
    block_M, block_N, block_K,
    dtype="float32",
    accum_dtype="float"
):
    """
    TileLang kernel for a single GEMM operation in float32.
    Computes: X @ W
    """
    @T.prim_func
    def single_gemm(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((K, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_K, block_N), dtype)
            
            output_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            T.clear(output_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(X[by * block_M, k * block_K], X_shared)
                T.copy(W[k * block_K, bx * block_N], W_shared)
                
                # Perform only ONE GEMM operation
                T.gemm(X_shared, W_shared, output_local)

            T.copy(output_local, Out[by * block_M, bx * block_N])

    return single_gemm

def main():
    # Define problem size
    M, K, N = 4096, 2048, 1024
    dtype = torch.float32
    device = "cuda"

    # Create random input tensors
    X = torch.randn(M, K, dtype=dtype, device=device)
    W = torch.randn(K, N, dtype=dtype, device=device)

    # Define tiling parameters
    block_M = 64
    block_N = 64
    block_K = 32
    
    # Instantiate and compile the simplified kernel
    kernel = single_gemm_kernel(M, N, K, block_M, block_N, block_K, dtype="float32")
    
    # Run the TileLang kernel
    output_tilelang = kernel(X, W)
    
    # --- Verification using PyTorch ---
    output_ref = X @ W
    
    print("Verifying correctness for a single float32 GEMM operation...")
    try:
        torch.testing.assert_close(output_tilelang, output_ref, rtol=1e-5, atol=1e-5)
        print("✅ Correctness check passed for single GEMM!")
    except AssertionError as e:
        print("❌ Correctness check FAILED for single GEMM!")
        print(e)


if __name__ == "__main__":
    main()
```

(venv) ➜  test git:(master) ✗ python tilelang_swiglu.py
Verifying correctness for a single float32 GEMM operation...
❌ Correctness check FAILED for single GEMM!
Tensor-likes are not close!

Mismatched elements: 4179519 / 4194304 (99.6%)
Greatest absolute difference: 0.2025909423828125 at index (599, 166) (up to 1e-05 allowed)
Greatest relative difference: 2067.269287109375 at index (2778, 409) (up to 1e-05 allowed)
When data type been switch to float16, 100% pass. The problem should be reproducible on your Tilelang environment.