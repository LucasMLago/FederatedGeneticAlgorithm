# Flower app

The federated learning app. Modules in [federatedgeneticalgorithm/](federatedgeneticalgorithm/):

| Module | What's in it |
|---|---|
| `client_app.py` | client train/eval, runs the per-client GA when enabled |
| `server_app.py` | FedAvg strategies, server-side HP search, telemetry hooks |
| `federated_genetic_algorithm.py` | server-side GA (population, one HP per round, elitism) |
| `genetic_algorithm.py`, `surrogate_model.py` | per-client GA and the shared surrogate |
| `federated_baselines.py` | random search and TPE for the broadcast protocol |
| `task.py` | datasets, models, Dirichlet partitioning |
| `config/config.py` | default config, overridden by the scenario YAMLs |
| `telemetry.py` | per-run CSV telemetry (see [artifacts/README.md](artifacts/README.md)) |
| `runner.py` | CLI that resolves a YAML + seed and calls `flwr run` |

For experiments use `make run CONFIG=... SEED=...` from the repo root instead of calling
`flwr run` yourself: the runner is what saves the config snapshot and the telemetry.
