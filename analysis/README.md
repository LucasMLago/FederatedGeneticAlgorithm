# Analysis scripts

All scripts read the run telemetry under `federatedgeneticalgorithm/artifacts/` (the matrix
summary CSV plus per-run directories) and are safe to re-run at any time; reports are
regenerated, never hand-edited.

| Script | Output | What it computes |
|---|---|---|
| `results_tables.py` | `results_tables.md` | Per-scenario stats grouped by family (CIFAR-main, FEMNIST, SmallCNN, α=0.1, failure-mode arms), round-level robustness events (severe drops >10pp, terminal collapse, rounds-to-X%), and the pairwise Mann–Whitney/Wilcoxon tests reported in the paper. |
| `fitness_bias.py` | `fitness_bias_report.md` | Reconstructs per-round GA traces (broadcast HP, fitness, generation) for every server-broadcast run; quantifies the cold-start bias (gen-0 rank of the first-evaluated individual, warm-start survival) and the trajectory-position bias (post-crash Δ credit, stale-elite staleness). |
| `paper_figures.py` | `figures/*.{pdf,png}` | Publication figures: the three-regime trade-off panel and the cold-start mechanism plot. |
| `aggregate_matrix.py` | stdout / optional md | Quick per-scenario mean/sd/CI table straight from the matrix CSV; safe mid-sweep. |
| `surrogate_ablation.py` | `surrogate_ablation_report.md`, `surrogate_ablation.png` | Paired N=5+5 surrogate ON/OFF ablation (peak, final, wall-time, crash rounds). |
