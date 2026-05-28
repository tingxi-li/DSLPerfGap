# Iterations

## rms_norm (tilelang) Optimization

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| baseline | Original T.serial kernel | 0.014x | 716ms | CORRECT |
| 1 | T.reduce + fp16 native + torch.empty + rsqrt | 11.06x | 0.903ms | CORRECT |
| 2 | threads=512, fewer fragments, fused store | 11.08x | 0.902ms | CORRECT |

---

### Iter 1: T.reduce + fp16 native + torch.empty + rsqrt
- Replaced T.serial sum-of-squares loop with T.reduce("sum")
- Changed kernel to accept fp16 tensors natively (eliminated .float() conversion overhead)
- Used torch.empty instead of torch.zeros for output allocation
- Used T.rsqrt instead of 1/T.sqrt for faster reciprocal square root
- Used float32 accumulator for precision in the reduction
- threads=256

