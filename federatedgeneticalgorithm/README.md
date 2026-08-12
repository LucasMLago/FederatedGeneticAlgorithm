# Flower application package

The federated learning application itself. Modules under
[`federatedgeneticalgorithm/`](federatedgeneticalgorithm/):

| Module | Role |
|---|---|
| `client_app.py` | client training/evaluation; runs the per-client GA when enabled |
| `server_app.py` | FedAvg server strategies + HP-search integration and telemetry hooks |
| `federated_genetic_algorithm.py` | server-side GA (population, round-robin evaluation, elitism) |
| `genetic_algorithm.py`, `surrogate_model.py` | per-client GA and the shared surrogate filter |
| `federated_baselines.py` | Random Search and TPE under the same broadcast protocol |
| `task.py` | datasets (CIFAR-10 / FEMNIST), models, Dirichlet partitioning |
| `config/config.py` | canonical configuration constants (overridden per scenario YAML) |
| `telemetry.py` | per-run CSV telemetry (see [`artifacts/README.md`](artifacts/README.md)) |
| `runner.py` | CLI: resolves a scenario YAML + seed and invokes `flwr run` |

Don't invoke `flwr run` directly for experiments; use `make run CONFIG=... SEED=...` from the
repository root (see the [root README](../README.md)), which goes through the runner so the
configuration snapshot and telemetry are recorded.
