# Analysis

The scripts read the telemetry in `federatedgeneticalgorithm/artifacts/` (the matrix summary CSV
and the per-run folders). The reports are written by the scripts, don't edit them by hand.

| Script | Writes | What it does |
|---|---|---|
| `results_tables.py` | `results_tables.md` | stats per scenario, severe drops (>10 pp in one round), rounds to X%, Mann-Whitney / Wilcoxon tests |
| `fitness_bias.py` | `fitness_bias_report.md` | rebuilds the round-by-round GA traces of the broadcast runs and measures the cold-start and trajectory-position biases |
| `paper_figures.py` | `figures/*.pdf`, `figures/*.png` | the paper figures |
| `client_agreement.py` | `client_agreement_report.md` | how often clients of the same round agree on optimizer or configuration, and how severe drops line up with harmful configurations |
| `aggregate_matrix.py` | stdout or a .md | quick mean/sd/CI per scenario straight from the matrix CSV, works mid-sweep |
| `surrogate_ablation.py` | `surrogate_ablation_report.md`, `surrogate_ablation.png` | surrogate on vs off, 5 seeds each |
| `drop_sensitivity.py` | `drop_sensitivity_report.md` | severe-drop counts at 5, 10 and 20 pp, largest drop per run, peak minus final |
| `multiple_testing.py` | `multiple_testing_report.md` | Holm and Benjamini-Hochberg adjustment of the ten-seed tests of Section V-A |
| `compute_budget.py` | `compute_budget_report.md` | client compute per run, split into federated training and search, in local-epoch equivalents |
| `ceiling_probe.py` | `ceiling_probe_report.md` | eight fixed SGD variants of the expert (seed 0) against the expert, to see whether anything nearby beats it |
| `lambda0_report.py` | `lambda0_report.md` | per-client GA without the drift penalty against the standard one, seeds 0-2: peak, severe drops, harmful updates |
| `pruned_space.py` | `pruned_space_report.md` | broadcast GA, RS and TPE on the pruned search space against the full space, seeds 0-4 |

The scripts read `matrix_summary_final.csv`, the runs the paper reports. To read another summary,
such as the `matrix_summary.csv` that `scripts/run_matrix.py` appends to, set `FGA_SUMMARY` to its
path. To regenerate the paper's reports and figures:

```bash
uv run --project federatedgeneticalgorithm python analysis/results_tables.py --markdown analysis/results_tables.md
uv run --project federatedgeneticalgorithm python analysis/client_agreement.py --markdown analysis/client_agreement_report.md
uv run --project federatedgeneticalgorithm python analysis/paper_figures.py
uv run --project federatedgeneticalgorithm python analysis/fitness_bias.py --markdown analysis/fitness_bias_report.md
uv run --project federatedgeneticalgorithm python analysis/surrogate_ablation.py
uv run --project federatedgeneticalgorithm python analysis/drop_sensitivity.py --markdown analysis/drop_sensitivity_report.md
uv run --project federatedgeneticalgorithm python analysis/multiple_testing.py --markdown analysis/multiple_testing_report.md
uv run --project federatedgeneticalgorithm python analysis/compute_budget.py --markdown analysis/compute_budget_report.md
uv run --project federatedgeneticalgorithm python analysis/ceiling_probe.py --markdown analysis/ceiling_probe_report.md
uv run --project federatedgeneticalgorithm python analysis/lambda0_report.py --markdown analysis/lambda0_report.md
uv run --project federatedgeneticalgorithm python analysis/pruned_space.py --markdown analysis/pruned_space_report.md
```

The ceiling probe and lambda-0 runs are listed in `matrix_summary_probes.csv` (`configs/probes/` holds the probe configs, written by
`scripts/make_probe_configs.py configs/probes`); the pruned-space runs go in `matrix_summary_pruned.csv`.
