# Analysis

The scripts read the telemetry in `federatedgeneticalgorithm/artifacts/` (the matrix summary CSV
and the per-run folders). The reports are written by the scripts, don't edit them by hand.

| Script | Writes | What it does |
|---|---|---|
| `results_tables.py` | `results_tables.md` | stats per scenario, severe drops (>10 pp in one round), rounds to X%, Mann-Whitney / Wilcoxon tests |
| `fitness_bias.py` | `fitness_bias_report.md` | rebuilds the round-by-round GA traces of the broadcast runs and measures the cold-start and trajectory-position biases |
| `paper_figures.py` | `figures/*.pdf`, `figures/*.png` | the paper figures |
| `aggregate_matrix.py` | stdout or a .md | quick mean/sd/CI per scenario straight from the matrix CSV, works mid-sweep |
| `surrogate_ablation.py` | `surrogate_ablation_report.md`, `surrogate_ablation.png` | surrogate on vs off, 5 seeds each |
