# Kernel Samples

Curated PyTorch kernel examples for use as few-shot prompts or code generation templates.
Each file follows a unified interface that a generator should learn to produce.

## Structure

```
<category>/
  primitive/    # Single operations (level1 / tritonbench)
  fused/        # 2-3 ops chained together (level2)
  model/        # Complete model architectures (level3)
```

Not every category has all three tiers.

## Unified Interface

Every sample exposes the same four functions:

```python
class Model(nn.Module):
    def __init__(self, ...): ...     # Constructor
    def forward(self, ...): ...      # Computation

def get_inputs():          # CPU input tensors
def get_init_inputs():     # Model constructor args
def get_test_inputs():     # CUDA-ready inputs
def run(*args):            # One-call entry point
```

## Categories

| Category | Primitive | Fused | Model | Total |
|----------|-----------|-------|-------|-------|
| matmul | 4 | 2 | 1 | 7 |
| conv | 3 | 2 | 2 | 7 |
| activation | 3 | - | - | 3 |
| normalization | 3 | - | - | 3 |
| attention | 1 | - | 1 | 2 |
| loss | 3 | - | - | 3 |
| reduction | 3 | - | - | 3 |
| pooling | 2 | - | - | 2 |
| cumulative | 2 | - | - | 2 |
| embedding | 1 | - | - | 1 |
| dropout | 1 | - | - | 1 |
| conv_transpose | - | 2 | - | 2 |
| model_cnn | - | - | 3 | 3 |
| model_transformer | - | - | 2 | 2 |
| model_rnn | - | - | 2 | 2 |

**Total: 43 samples across 15 categories and 3 difficulty tiers.**

## Provenance

Selected from KernelBench (level1/level2/level3) and TritonBench.
Original kernel IDs are in the file headers.
