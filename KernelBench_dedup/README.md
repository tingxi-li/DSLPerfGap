# KernelBench — Comprehensive Kernel Dataset

## Sources
- **KernelBench** (github.com/ScalingIntelligence/KernelBench): L1/L2/L3
- **TritonBench** (github.com/meta-pytorch/tritonbench): Triton operators
- **ViperBench**: 22 existing kernels (not duplicated here)

## Kernel Counts
| Category | Count |
|----------|-------|
| Level 1 — Individual Ops | 100 |
| Level 2 — Fused Op Chains | 100 |
| Level 3 — Full Models | 50 |
| TritonBench Operators | 49 |
| **Total** | **299** |

## Directory Structure
```
KernelBench/
├── level1/           # Individual ops from KernelBench
│   ├── 1_square_matrix_multiplication/
│   │   └── pytorch_impl.py
│   └── ...
├── level2/           # Fused op chains from KernelBench
│   ├── 1_conv2d_relu_biasadd/
│   │   └── pytorch_impl.py
│   └── ...
├── level3/           # Full models from KernelBench
│   ├── 1_mlp/
│   │   └── pytorch_impl.py
│   └── ...
└── tritonbench/      # TritonBench operators (with Triton code)
    ├── flash_attention/
    │   ├── operator.py
    │   └── ...
    └── ...
```

## File Format

### KernelBench (level1/level2/level3)
Each `pytorch_impl.py` contains:
- `Model(nn.Module)` class with `forward()` method
- `get_inputs()` — returns list of input tensors
- `get_init_inputs()` — returns constructor arguments for Model

### TritonBench
Each operator directory contains the original tritonbench source files
including Triton kernel implementations and PyTorch baselines.

## Usage with ViperBench eval_comprehensive.py
These kernels can be integrated into the evaluation harness by:
1. Importing the Model class from pytorch_impl.py
2. Using get_inputs() and get_init_inputs() to create test cases
3. Running through the adaptive tolerance metric
