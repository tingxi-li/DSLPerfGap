# Cross-architecture comparison data

Auto-generated from the per-GPU result CSVs under `experiments/results/<gpu>/` —
**numbers are not hand-typed**, so they cannot drift from the underlying data.

| File | What it is |
|---|---|
| `cross_arch_data.json` | machine-readable merged per-GPU numbers (source of truth) |
| `gen_cross_arch_tables.py` | regenerates `cross_arch_data.json` + a LaTeX table dump (`cross_arch.tex`) from the per-GPU CSVs |
| `tuned_comparison.csv` | tuned-vs-untuned library-efficiency comparison across GPUs |

The median-`E_lib`-per-category generalization summary derived from this data is
reproduced in the top-level [`README.md`](../../../README.md) under
"Cross-Architecture Generalization".

## Regenerate after new data

```bash
python experiments/results/cross_arch/gen_cross_arch_tables.py   # rebuilds the JSON + cross_arch.tex here
```

`---` cells in the tables are genuine OOM-skips on the 40 GB A100 (5×5 / 7×7 TileLang conv).
