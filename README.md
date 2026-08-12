# Coupling regimes in federated hyperparameter search

Code and data for the paper "How Much Should Clients Agree on Hyperparameters? Characterizing
Coupling Regimes in Federated Hyperparameter Search".

We compare hyperparameter search designs for federated learning that differ in how much the
clients share their HP choices within a round: per-client GA (zero coupling), per-client GA with a
shared surrogate (medium) and a server-side GA that sends one config per round to everyone (high).
There are also fixed-HP baselines and two non-GA searchers (random search, TPE) using the same
broadcast protocol. Built on Flower + PyTorch, with CIFAR-10 and FEMNIST.

## Quick start

```bash
make install
make smoke                                               # 2 rounds on CPU, ~6 min
make run CONFIG=configs/ga_broadcast_cifar.yaml SEED=0   # one real run
```

Each scenario is a YAML file, see [configs/README.md](configs/README.md). To run several seeds:

```bash
federatedgeneticalgorithm/.venv/bin/python scripts/run_matrix.py \
    --configs configs/ga_broadcast_cifar.yaml --seeds 0 1 2 3 4 \
    --federation local-simulation-gpu --min-eval-rounds 20
```

Every finished run adds a row to `federatedgeneticalgorithm/artifacts/matrix_summary.csv`. Pairs
that are already there get skipped, so a sweep can be restarted. Call the venv python directly
instead of `uv run`: Ray passes the uv wrapper on to its workers and they pick up the wrong project.

## Tables and figures

All of them come from the telemetry in the repo:

```bash
uv run --project federatedgeneticalgorithm python analysis/results_tables.py
uv run --project federatedgeneticalgorithm python analysis/fitness_bias.py
uv run --project federatedgeneticalgorithm python analysis/paper_figures.py
```

## Layout

```
configs/                       scenario YAMLs
scripts/run_matrix.py          config x seed sweeps
federatedgeneticalgorithm/
  federatedgeneticalgorithm/   Flower app (client, server, GA, surrogate, baselines, telemetry)
  artifacts/                   matrix_summary.csv and per-run telemetry
analysis/                      analysis scripts and the reports they write
```
