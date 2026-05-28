#!/usr/bin/env python3
"""
Generate KernelBench-format wrappers from ViperBench kernels for AKO4ALL.

Usage:
    python prepare_kernel.py <kernel_name> <impl>

Example:
    python prepare_kernel.py matmul triton
    python prepare_kernel.py layer_norm tilelang

This will:
  1. Read ViperBench/<kernel>/pytorch_impl.py (reference)
  2. Read ViperBench/<kernel>/<impl>_impl.py (solution to optimize)
  3. Generate input/reference.py and solution/kernel.py in KernelBench format
  4. Generate scripts/bench.sh
"""
import argparse
import sys
from pathlib import Path

VIPER_DIR = Path(__file__).parent.parent / "ViperBench"
AKO_DIR = Path(__file__).parent

KERNEL_CONFIGS = {
    "matmul": {
        "fn": "matmul",
        "get_inputs": (
            "def get_inputs():\n"
            "    a = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)\n"
            "    b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)\n"
            "    return [a, b]\n"
        ),
        "precision": "float16",
    },
    "argmax": {
        "fn": "argmax",
        "get_inputs": (
            "def get_inputs():\n"
            "    return [torch.randn(8192, 32768, device='cuda', dtype=torch.float16), 1]\n"
        ),
        "precision": "float16",
    },
    "max_reduction": {
        "fn": "max_reduction",
        "get_inputs": (
            "def get_inputs():\n"
            "    return [torch.randn(8192, 32768, device='cuda', dtype=torch.float16), 1]\n"
        ),
        "precision": "float16",
    },
    "conv2d": {
        "fn": "conv2d",
        "get_inputs": (
            "def get_inputs():\n"
            "    x = torch.randn(32, 256, 128, 128, device='cuda', dtype=torch.float16)\n"
            "    w = torch.randn(256, 256, 3, 3, device='cuda', dtype=torch.float16)\n"
            "    return [x, w]\n"
        ),
        "precision": "float16",
    },
    "linear_activation": {
        "fn": "kernel_ff",
        "get_inputs": (
            "def get_inputs():\n"
            "    x = torch.randn(1, 2048, 4096, device='cuda', dtype=torch.float16)\n"
            "    w1 = torch.randn(16384, 4096, device='cuda', dtype=torch.float16)\n"
            "    w3 = torch.randn(16384, 4096, device='cuda', dtype=torch.float16)\n"
            "    rms_w = torch.randn(4096, device='cuda', dtype=torch.float16)\n"
            "    return [x, w1, w3, rms_w]\n"
        ),
        "precision": "float16",
    },
    "layer_norm": {
        "fn": "layer_norm",
        "get_inputs": (
            "def get_inputs():\n"
            "    x = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)\n"
            "    weight = torch.randn(8192, device='cuda', dtype=torch.bfloat16)\n"
            "    bias = torch.randn(8192, device='cuda', dtype=torch.bfloat16)\n"
            "    return [x, weight, bias]\n"
        ),
        "precision": "bfloat16",
    },
    "rms_norm": {
        "fn": "rms_norm",
        "get_inputs": (
            "def get_inputs():\n"
            "    x = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)\n"
            "    normalized_shape = (8192,)\n"
            "    weight = torch.randn(8192, device='cuda', dtype=torch.float16)\n"
            "    return [x, normalized_shape, weight]\n"
        ),
        "precision": "float16",
    },
    "mean_reduction": {
        "fn": "mean_reduction",
        "get_inputs": (
            "def get_inputs():\n"
            "    return [torch.randn(8192, 32768, device='cuda', dtype=torch.float32), 1]\n"
        ),
        "precision": "float32",
    },
    "relu": {
        "fn": "relu",
        "get_inputs": (
            "def get_inputs():\n"
            "    return [torch.randn(16384, 16384, device='cuda', dtype=torch.float16)]\n"
        ),
        "precision": "float16",
    },
    "softmax": {
        "fn": "softmax",
        "get_inputs": (
            "def get_inputs():\n"
            "    return [torch.randn(4096, 32768, device='cuda', dtype=torch.float16)]\n"
        ),
        "precision": "float16",
    },
    "add": {
        "fn": "add",
        "get_inputs": (
            "def get_inputs():\n"
            "    M = 64 * 1024 * 1024\n"
            "    return [torch.randn(M, device='cuda', dtype=torch.float16),\n"
            "            torch.randn(M, device='cuda', dtype=torch.float16)]\n"
        ),
        "precision": "float16",
    },
    "mul": {
        "fn": "mul",
        "get_inputs": (
            "def get_inputs():\n"
            "    M = 64 * 1024 * 1024\n"
            "    return [torch.randn(M, device='cuda', dtype=torch.float16)]\n"
        ),
        "precision": "float16",
    },
    "embedding": {
        "fn": "embedding",
        "get_inputs": (
            "def get_inputs():\n"
            "    ids = torch.randint(0, 131072, (131072,), device='cuda', dtype=torch.int32)\n"
            "    w = torch.randn(131072, 1024, device='cuda', dtype=torch.float16)\n"
            "    return [ids, w, 0, 131072,\n"
            "            torch.zeros(131072, 1024, device='cuda', dtype=torch.float16)]\n"
        ),
        "precision": "float16",
    },
    "logsumexp": {
        "fn": "logsumexp",
        "get_inputs": (
            "def get_inputs():\n"
            "    return [torch.randn(8192, 16384, device='cuda', dtype=torch.float32)]\n"
        ),
        "precision": "float32",
    },
    "log_softmax": {
        "fn": "log_softmax",
        "get_inputs": (
            "def get_inputs():\n"
            "    return [torch.randn(4096, 32768, device='cuda', dtype=torch.float16)]\n"
        ),
        "precision": "float16",
    },
    "swiglu": {
        "fn": "swiglu",
        "get_inputs": (
            "def get_inputs():\n"
            "    return [torch.randn(4096, 32768, device='cuda', dtype=torch.float16)]\n"
        ),
        "precision": "float16",
    },
    "matrix_transpose": {
        "fn": "matrix_transpose",
        "get_inputs": (
            "def get_inputs():\n"
            "    return [torch.randn(16384, 16384, device='cuda', dtype=torch.float16)]\n"
        ),
        "precision": "float16",
    },
    "batched_matmul": {
        "fn": "batched_matmul",
        "get_inputs": (
            "def get_inputs():\n"
            "    A = torch.randn(128, 2048, device='cuda', dtype=torch.float16)\n"
            "    B = torch.randn(128, 2048, 2048, device='cuda', dtype=torch.float16)\n"
            "    return [A, B]\n"
        ),
        "precision": "float16",
    },
    "index_select": {
        "fn": "index_select",
        "get_inputs": (
            "def get_inputs():\n"
            "    output = torch.empty(4096, 2048, device='cuda', dtype=torch.float16)\n"
            "    source = torch.randn(65536, 2048, device='cuda', dtype=torch.float16)\n"
            "    index = torch.randint(0, 65536, (4096,), device='cuda', dtype=torch.int64)\n"
            "    return [output, source, index]\n"
        ),
        "precision": "float16",
    },
    "attention": {
        "fn": "attention_fwd",
        "get_inputs": (
            "def get_inputs():\n"
            "    q = torch.randn(8, 32, 2048, 64, device='cuda', dtype=torch.float32)\n"
            "    k = torch.randn(8, 32, 2048, 64, device='cuda', dtype=torch.float32)\n"
            "    v = torch.randn(8, 32, 2048, 64, device='cuda', dtype=torch.float32)\n"
            "    return [q, k, v]\n"
        ),
        "precision": "float32",
    },
    "leaky_relu": {
        "fn": "leaky_relu",
        "get_inputs": (
            "def get_inputs():\n"
            "    a = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)\n"
            "    b = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)\n"
            "    return [a, b, 'leaky_relu']\n"
        ),
        "precision": "float16",
    },
    "cross_entropy": {
        "fn": "cross_entropy_fwd",
        "get_inputs": (
            "def get_inputs():\n"
            "    logits = torch.randn(4096, 32768, device='cuda', dtype=torch.float32)\n"
            "    labels = torch.randint(0, 32768, (4096,), device='cuda', dtype=torch.int64)\n"
            "    return [logits, labels, 0.0, 1.0, 0.0, -100, 32768, 0, 1024, False, False]\n"
        ),
        "precision": "float32",
    },
}


def read_impl_source(kernel_name, impl_type):
    if impl_type == "triton":
        path = VIPER_DIR / kernel_name / "triton_impl.py"
    elif impl_type == "tilelang":
        path = VIPER_DIR / kernel_name / "tilelang_impl.py"
    elif impl_type == "pytorch":
        path = VIPER_DIR / kernel_name / "pytorch_impl.py"
    else:
        raise ValueError(f"Unknown impl type: {impl_type}")
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    return path.read_text()


def generate_kb_wrapper(source_code, config):
    fn_name = config["fn"]
    get_inputs = config["get_inputs"]
    return (
        f"import torch\nimport torch.nn as nn\n\n"
        f"# --- Original implementation inlined below ---\n"
        f"{source_code}\n"
        f"# --- End original implementation ---\n\n\n"
        f"class Model(nn.Module):\n"
        f"    def __init__(self):\n"
        f"        super().__init__()\n\n"
        f"    def forward(self, *args):\n"
        f"        return {fn_name}(*args)\n\n\n"
        f"{get_inputs}\n"
        f"def get_init_inputs():\n"
        f"    return []\n"
    )


def setup_ako(kernel_name, impl_type):
    if kernel_name not in KERNEL_CONFIGS:
        print(f"ERROR: Unknown kernel '{kernel_name}'.")
        print(f"Available: {', '.join(sorted(KERNEL_CONFIGS.keys()))}")
        sys.exit(1)

    config = KERNEL_CONFIGS[kernel_name]
    precision = config["precision"]

    pytorch_source = read_impl_source(kernel_name, "pytorch")
    impl_source = read_impl_source(kernel_name, impl_type)

    ref_wrapper = generate_kb_wrapper(pytorch_source, config)
    sol_wrapper = generate_kb_wrapper(impl_source, config)

    input_dir = AKO_DIR / "input"
    solution_dir = AKO_DIR / "solution"
    scripts_dir = AKO_DIR / "scripts"
    input_dir.mkdir(exist_ok=True)
    solution_dir.mkdir(exist_ok=True)
    scripts_dir.mkdir(exist_ok=True)

    # Clear old files
    for f in input_dir.glob("*.py"):
        f.unlink()
    for f in solution_dir.glob("*.py"):
        f.unlink()

    (input_dir / "reference.py").write_text(ref_wrapper)
    (solution_dir / "kernel.py").write_text(sol_wrapper)
    (input_dir / f"{impl_type}_impl.py").write_text(impl_source)

    # Generate bench.sh
    backend = impl_type if impl_type in ("triton", "tilelang") else "cuda"
    bench_cmd = (
        f"python ../bench/kernelbench/bench.py "
        f"--ref ../input/reference.py "
        f"--solution ../solution/kernel.py "
        f"--precision {precision} "
        f"--backend {backend} "
        f"--verbose "
        f"2>&1 | tee _bench_output.txt"
    )
    wrapper_template = (AKO_DIR / "bench-wrapper.sh").read_text()
    bench_sh = wrapper_template.replace("{{BENCH_COMMAND}}", bench_cmd)
    bench_sh_path = scripts_dir / "bench.sh"
    bench_sh_path.write_text(bench_sh)
    bench_sh_path.chmod(0o755)

    # Reset ITERATIONS.md
    (AKO_DIR / "ITERATIONS.md").write_text(
        f"# Iterations\n\n"
        f"## {kernel_name} ({impl_type}) Optimization\n\n"
        f"| Iter | Title | Speedup(mean) | Runtime(mean) | Status |\n"
        f"|------|-------|---------|--------------|--------|\n\n---\n\n"
    )

    print(f"AKO4ALL configured for: {kernel_name} ({impl_type})")
    print(f"  Reference:  input/reference.py (PyTorch)")
    print(f"  Solution:   solution/kernel.py ({impl_type})")
    print(f"  Precision:  {precision}")
    print(f"  Benchmark:  scripts/bench.sh")
    print(f"\nNext steps:")
    print(f"  cd AKO4ALL")
    print(f"  bash scripts/bench.sh baseline  # verify baseline")
    print(f"  # Then follow TASK.md optimization loop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AKO4ALL for a ViperBench kernel")
    parser.add_argument("kernel", help="Kernel name (e.g., matmul, layer_norm)")
    parser.add_argument("impl", choices=["triton", "tilelang"],
                        help="Implementation to optimize")
    args = parser.parse_args()
    setup_ako(args.kernel, args.impl)
