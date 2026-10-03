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

## `matrix_summary_valfit.csv`, `matrix_summary_final.csv`, `expert_position.csv`

- `matrix_summary_valfit.csv`: the runs redone after server-side search moved its fitness to the
  held-out validation split and the per-client GA got an evolving population, plus the ablations and
  the 40-round runs. Same columns as above. Its `git_sha` column shows the server checkout (61b6d54):
  the code was copied there with rsync before it was committed, and it is the code merged into `main`
  after that commit. The rows tagged `week-*` (seeds 5 to 9 of the main scenario, FedEx, the
  long-evaluation surrogate and the expert with 40 rounds) show d5a03e7 for the same reason: FedEx
  was copied over before its commit.
- `matrix_summary_final.csv`: the rows the paper reports, i.e. the fixed-configuration rows of
  `matrix_summary.csv` (no search, so the fix does not touch them) plus every `ok` row of
  `matrix_summary_valfit.csv`. Point the analysis scripts at it with `FGA_SUMMARY`.
- `expert_position.csv`: output of `scripts/expert_position_checkpoint.py`, the common-checkpoint
  test of the broadcast GA's generation 0.

## `runs/<run_id>/`

Written during the run:

| File | Contents | Used by |
|---|---|---|
| `server_aggregated_rounds.csv` | aggregated train/eval metrics per round (the eval-acc curves) | `results_tables.py`, `paper_figures.py` |
| `client_round_metrics.csv` | per-client metrics per round and the HPs each client used | `client_agreement.py` |
| `config.yaml`, `resolved_config.json` | the config the run actually used | reproducing a run |
| `run_metadata.json` | run id, git SHA, tag, resolved config | provenance |
| `partition_distribution.json` | class histogram per client after the Dirichlet split | checking the partitions |
| `ga_candidates.csv`, `ga_state/` | per-client GA populations and evaluated candidates (per-client scenarios only) | GA inspection |
| `hp_pool_client_<id>.pkl` | each client's own surrogate pool (surrogate runs without the shared pool) | surrogate inspection |
| `fedex_rounds.csv` | FedEx per round: weighted validation error, baseline, entropy and most likely HP of the distribution | FedEx inspection |

The server-side GA trace (broadcast HP and fitness per round) only goes to the app log, not to
per-run files. `analysis/fitness_bias.py` rebuilds it by matching log lines to each run's time
window in `matrix_summary.csv`. The log isn't committed; `broadcast_traces.txt` here is the
committed excerpt with the `[FedGA]`/`[HPSearch]` lines, and the script uses it when the log is
missing.
