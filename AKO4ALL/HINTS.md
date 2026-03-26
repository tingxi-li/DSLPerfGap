# Hints

- Before Iter 1, run `ncu` on the baseline kernel to guide the first direction.
- If 3 consecutive iterations show no improvement, run `ncu` to re-profile, use WebSearch for new ideas, and review `ITERATIONS.md` for patterns. Plan before continuing.
- Focus on large-input performance — small-input overhead is less important.
- For Triton kernels: tune block sizes, num_warps, num_stages. Try autotuning with @triton.autotune.
- For TileLang kernels: use T.Pipelined with num_stages for memory latency hiding, T.use_swizzle for L2 locality, T.gemm for matmul-heavy ops. Avoid T.serial loops for reductions — use T.reduce instead.
- Do not install any packages.
- You may rewrite the kernel in a different language (e.g., Triton → CUDA, TileLang → Triton) if it achieves better performance.
