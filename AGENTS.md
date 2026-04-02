# Agent Guidelines

## Kernel Porting Agent
When porting a kernel to TileLang:
1. Read the PyTorch reference and Triton implementation fully before writing any code
2. Use `@tilelang.jit` decorator (never raw kernel caching)
3. TileLang must stay TileLang — no falling back to Triton or cuDNN
4. Follow the retry strategy in CLAUDE.md (up to 20 attempts per kernel)
5. Log results to `tests/results/<kernel>.json`

## Code Review Agent
When reviewing TileLang implementations:
- Verify numerical correctness tolerances match CLAUDE.md table
- Check for missing `T.clear()` before accumulations
- Ensure block sizes divide tensor dimensions or padding is used
- Confirm `T.copy` direction is correct (global->shared, fragment->global)

## Benchmarking Agent
When running benchmarks:
- Always compare TileLang vs Triton vs PyTorch
- Use consistent input shapes from `kernel_input_shapes.html`
- Report throughput in GFLOPS or GB/s as appropriate
- Save results to `tests/results/`

## General Rules
- The kernel directory is `ViperBench/` (not `newBench/` despite CLAUDE.md references)
- Each kernel subdirectory contains: PyTorch ref, Triton impl, `tilelang_impl.py`, `test_kernel.py`
- Shared test utilities live in `ViperBench/test_utils.py`
- Project-level test harness: `test_harness.py` at project root
