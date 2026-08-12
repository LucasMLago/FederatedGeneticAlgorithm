# Run artifacts

Everything here is written by the runner and the telemetry code while the runs execute, and read by
the scripts in `analysis/`. Nothing is edited by hand.

## `matrix_summary.csv`

One row per finished `(config, seed)` run.

| Column | Meaning |
|---|---|
| `config`, `seed` | scenario YAML and seed. Also the resume key: `ok` rows are skipped on relaunch, `failed` ones run again |
| `scenario_name` | the `name:` field of the config |
| `run_id` | timestamp id, same as the folder under `runs/` |
| `status`, `returncode` | `ok` needs exit code 0 and at least `--min-eval-rounds` eval rounds in the telemetry (catches runs cut short, e.g. by memory pressure) |
| `peak_eval_acc`, `final_eval_acc`, `num_eval_rounds` | read from the run's server telemetry |
| `wall_seconds` | total run time |
| `git_sha`, `tag`, `started_at`, `finished_at` | where the run came from |

## `runs/<run_id>/`

Written during the run:

| File | Contents | Used by |
|---|---|---|
| `server_aggregated_rounds.csv` | aggregated train/eval metrics per round (the eval-acc curves) | `results_tables.py`, `paper_figures.py` |
| `client_round_metrics.csv` | per-client metrics per round and the HPs each client used | manual inspection |
| `config.yaml`, `resolved_config.json` | the config the run actually used | reproducing a run |
| `run_metadata.json` | run id, git SHA, tag, resolved config | provenance |
| `partition_distribution.json` | class histogram per client after the Dirichlet split | checking the partitions |
| `ga_candidates.csv`, `ga_state/` | per-client GA populations and evaluated candidates (per-client scenarios only) | GA inspection |

The server-side GA trace (broadcast HP and fitness per round) only goes to the app log, not to
per-run files. `analysis/fitness_bias.py` rebuilds it by matching log lines to each run's time
window in `matrix_summary.csv`. The log isn't committed; `broadcast_traces.txt` here is the
committed excerpt with the `[FedGA]`/`[HPSearch]` lines, and the script uses it when the log is
missing.
