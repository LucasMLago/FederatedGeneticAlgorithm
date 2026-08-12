# Run artifacts

This directory is the experimental record. Nothing here is hand-edited; every file is written by
the runner/telemetry at run time and consumed by the scripts in `analysis/`.

## `matrix_summary.csv`

One row per completed `(config, seed)` run: the index of the whole experiment matrix. Columns:

| Column | Meaning |
|---|---|
| `config`, `seed` | which scenario YAML and seed the run used (also the resume key: `ok` rows are skipped on relaunch, `failed` rows retried) |
| `scenario_name` | the `name:` field of the config |
| `run_id` | timestamp id; points to the run's directory under `runs/` |
| `status`, `returncode` | `ok` requires exit 0 **and** ≥ `--min-eval-rounds` parsed eval rounds (guards against runs shredded mid-way by host memory pressure) |
| `peak_eval_acc`, `final_eval_acc`, `num_eval_rounds` | headline metrics parsed from the run's server telemetry |
| `wall_seconds` | end-to-end run duration |
| `git_sha`, `tag`, `started_at`, `finished_at` | provenance |

## `runs/<run_id>/`

Per-run telemetry, written live during the run:

| File | Contents | Used by |
|---|---|---|
| `server_aggregated_rounds.csv` | per-round aggregated train/eval metrics (the eval-acc curves) | `results_tables.py`, `paper_figures.py` |
| `client_round_metrics.csv` | per-client, per-round local metrics and the HPs each client actually used | per-client inspection |
| `config.yaml` + `resolved_config.json` | the exact configuration snapshot the run executed | reproducibility |
| `run_metadata.json` | run id, git SHA, tag, full resolved config | provenance |
| `partition_distribution.json` | class histogram per client under the Dirichlet split | partition sanity checks |
| `ga_candidates.csv`, `ga_state/` | per-client GA populations and evaluated candidates (per-client scenarios only) | GA inspection |

The per-round *server-side GA* trace (broadcast HP + fitness per round) lives in the application
log rather than per-run files; `analysis/fitness_bias.py` reconstructs it by matching log lines to
each run's time window from `matrix_summary.csv`. The log itself is not committed;
`broadcast_traces.txt` (in this directory) is the committed excerpt of exactly those
`[FedGA]`/`[HPSearch]` lines, and the script falls back to it automatically.
