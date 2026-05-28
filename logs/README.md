# logs/ — rebuttal campaign trajectory (archived)

These are the working documents from the ASE 2026 #4134 rebuttal effort, moved here so the repo
root carries only the single first-class artifact (`../REBUTTAL.md`) and the forward-looking
`../REVISION_TODO.md`. Nothing here was deleted; every file is the full original.

> The **canonical, submit-ready rebuttal text now lives in `../REBUTTAL.md` §1** — the copy inside
> `REBUTTAL_GAME_PLAN.md` below is the historical working draft.

| File | What it is |
|---|---|
| `REBUTTAL_GAME_PLAN.md` | Master strategy: the "how concerns sort" bins (A/B/C), the working rebuttal draft, the claim→backing→status coordination table, and the integrity gates. |
| `REVIEWER_WEAKNESS_ANALYSIS.md` | Full per-weakness evaluation (W1–W13 + minor), with severity and disposition for each. |
| `ADDITIONAL_EXPERIMENTS_PLAN.md` | The experiment plan — what to run, priorities, expected payoff per reviewer ask. |
| `REBUTTAL_EXPERIMENT_PROTOCOLS.md` | Do-now cheap-win protocols (per-kernel table, conv sweep, tuning definition) with filled result tables. |
| `REBUTTAL_PROTOCOLS_CRITICAL.md` | Detailed protocols for the hard experiments: ncu counter suite, FP32 root-cause, RC2b/RC3/RC4 isolations, 2nd-GPU study. |
| `RIGOR_AUDIT.md` | 3-agent synthesis of "what reviewers asked on profiling/benchmarking rigor"; concluded Reviewer1's two load-bearing asks — clocks (#A) and counters (#B) — are both done. |
| `CONSISTENCY_AUDIT.md` | 5-way pre-submission audit of rebuttal ↔ paper ↔ code; the source of the `../REVISION_TODO.md` items (incl. the N1–N3 internal findings). |

**Measured evidence** referenced throughout lives outside this folder, under
`../experiments/results/<gpu>/` and `../ViperBench/results/profile.csv`.
