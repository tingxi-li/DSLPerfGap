# MACHINE_SETUP.md — per-GPU software stacks + pre-move checklist

This repo is run on **multiple GPUs** (results auto-namespace under
`experiments/results/<gpu_slug>/` via `_harness.device_slug()`). The software stack
is **not** the same on every machine — `requirements.txt` holds the x86 pins used for
the Ada/A100/H100 numbers, but aarch64 (GH200) needs different pins. Record per-machine
stacks here so cross-architecture numbers can be reconciled, and follow the
**pre-move checklist** before releasing any GPU so its evidence is not lost.

> Companion docs: `A100_H100_RUNBOOK.md` (turnkey replay sequence) and
> `../PIVOT_FRAMING.md` (the paper's current framing + the new experiments to build).

## Per-GPU software stacks

| GPU (slug) | Arch | torch | triton | tilelang | Notes |
|---|---|---|---|---|---|
| RTX 4000 Ada / A100-SXM4-40GB / A100-PCIE-40GB / H100-80GB-HBM3 | x86_64 | 2.8.0+cu126 | 3.4.0 | 0.1.6.post1 | The `requirements.txt` pins; cuDNN 9.1.0.2, Nsight Compute 2024.3.2.0. **Paper-canonical stack.** |
| **NVIDIA GH200 480GB** | **aarch64** (Grace Hopper, sm_90) | **2.7.0** | **3.3.0** | **0.1.11** | TileLang **not** preinstalled; pins below. PyTorch is very fast on HBM3 memory-bound ops, so DSL-vs-PyTorch gaps differ from Ada — always re-baseline. |

### aarch64 / GH200 install path (not in `requirements.txt`)

```bash
pip install tilelang==0.1.11
pip install "numpy<2"                 # numpy 2.x breaks ml_dtypes (_ARRAY_API) -> import crash
pip install "apache-tvm-ffi==0.1.11"  # 0.1.12 aborts: double __ffi_repr__ type registration
```

Gotcha: TileLang reads `prim_func` source via `inspect.getsourcelines`, so kernels must
live in a real `.py` file — `python -c "<inline kernel>"` fails with "could not get source
code". Naive `T.serial` kernels (e.g. `ViperBench/layer_norm/tilelang_impl.py`) compile
*very* slowly on 0.1.11 (≈146 s for LayerNorm 8192²) and the `(16,2048)` correctness shape
can stall — budget for long compiles or skip that edge shape.

## Pre-move checklist (run BEFORE releasing a GPU)

When the source box is the GPU of interest (e.g. the GH200 right now), its evidence must
be captured and committed or it is lost on the move. In order:

1. **Capture results on this GPU.** Run the suite — see `A100_H100_RUNBOOK.md` §(0).
   The locked-clock pass (`repro/lock_clocks.sh`, `repro/run_pipeline.sh`) needs `sudo`.
   Override the (dead) default venv with `PYTHON=$(which python3)` if `dslperf-venv` is absent.
2. **Tuning sweep (for headline `*_tuned` numbers).** `tuning_cache.json` has **no** key for
   this GPU unless you sweep, so `*_tuned == *` until then:
   `cd ViperBench && python -m tuning.sweep --all` then re-run `benchmark_tuned.py`.
3. **Record the sustained clock.** Add this GPU's verified `gr,mem` clock row to
   `repro/lock_clocks.sh` and `experiments/exp_significance.py` (Ada/A100/H100 rows exist;
   GH200/Hopper is **not** yet recorded — discovery mode must find it).
4. **Commit everything.** `git add experiments/results/<slug>/ ViperBench/results/*.json`
   and commit. Check `git status` is clean for results before powering down the box.
5. **Confirm the `gpu_slug`** (`python -c "import torch;print(torch.cuda.get_device_name(0))"`)
   so the target's slug is known (idempotent re-run only if the slug matches an existing dir).

## Known state (2026-06-25)

- `experiments/results/` has committed dirs for: `NVIDIA_RTX_4000_Ada_Generation`,
  `A100-SXM4-40GB`, `A100-PCIE-40GB`, `H100_80GB_HBM3`. **No GH200 dir yet.**
- `tuning_cache.json` has keys for those 4 archs (143 keys). **No GH200 key.**
- The GH200 currently has **uncommitted** `ViperBench/results/*_tilelang.json` (round-2
  optimization results) — commit before moving (checklist step 4).
