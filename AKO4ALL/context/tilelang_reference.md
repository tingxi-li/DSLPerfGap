# TileLang Optimization Reference

## Key Primitives
- `T.Pipelined(N, num_stages=k)` — software pipelined loop for memory latency hiding
- `T.use_swizzle(panel_size=10)` — L2 cache locality hint
- `T.gemm(A, B, C)` — tile GEMM using tensor cores
- `T.reduce(src, dst, op)` — parallel tile reduction (NEVER use T.serial for reductions)
- `T.alloc_shared(shape, dtype)` — shared memory (fast, limited)
- `T.alloc_fragment(shape, dtype)` — register fragment
- `T.copy(src, dst)` — async tile copy
- `T.clear(buf)` — zero a buffer

## Common Performance Issues
1. Using T.serial() for reductions instead of T.reduce() — causes sequential execution
2. Small block sizes — underutilize GPU parallelism
3. Missing T.Pipelined — no memory latency hiding
4. Missing T.use_swizzle — poor L2 cache hit rate
5. Fragment allocation too large — register pressure causes spills
6. Not using T.gemm for matmul-heavy kernels — manual loops are much slower
7. threads= in T.Kernel must be a multiple of 32 (warp size)

## Best Practices
- Always use `@tilelang.jit` decorator (not `tilelang.compile`)
- Use `T.gemm` for matmul operations — leverages tensor cores
- Use `T.Pipelined` with `num_stages=3` for K-loop in GEMM patterns
- Use float32 accumulator (`T.alloc_fragment(..., "float32")`) for fp16 inputs
- Pad dimensions to multiples of block sizes
- For reductions: use `T.reduce(src, dst, op, dim=0)` instead of serial loops
