# Cross-architecture comparison artifacts (Ada sm_89 vs A100 sm_80)

Auto-generated from the per-GPU result CSVs — **numbers are not hand-typed**, so
they cannot drift from the data (unlike the hand-entered tables elsewhere).

| File | What it is |
|---|---|
| `cross_arch_data.json` | machine-readable merged Ada/A100 numbers (source of truth for the figure) |
| `../../../paper-latex-project/tex/cross_arch.tex` | 4 booktabs tables, ready to `\input` |
| `../../../paper-latex-project/figures/gen_cross_arch.py` | figure generator (reads the JSON) |
| `../../../paper-latex-project/figures/cross_arch_efficiency.pdf` | the figure (fig:xarch) |

## Tables (labels)
- `tab:xarch:summary`  — median E_lib per category, both DSLs, Ada vs A100 (generalization headline)
- `tab:xarch:rootcause`— RC0/RC3/RC4/FP32 reproduce across both GPUs
- `tab:xarch:autotune` — Triton matmul plain-vs-autotuned recovery, both GPUs
- `tab:xarch:conv`     — conv filter sweep (1x1..7x7) E_lib + Triton n_spills, both GPUs

## To include in the paper
```latex
\input{tex/cross_arch}                                  % the four tables
\includegraphics[width=\linewidth]{figures/cross_arch_efficiency.pdf}  % fig:xarch
```

## To regenerate after new data (e.g. tuned rows, or an H100 run)
```bash
source /venv/main/bin/activate && source /workspace/.env
python experiments/results/cross_arch/gen_cross_arch_tables.py   # rebuilds tex + json
cd paper-latex-project/figures && python gen_cross_arch.py   # rebuilds the figure
```
`---` cells = genuine OOM-skips on the 40 GB A100 (5x5/7x7 TileLang conv); the
RTX-4000 Ada counterpart CSVs are the cross-check baseline.
