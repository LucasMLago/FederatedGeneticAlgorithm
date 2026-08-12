# Coupling Regimes in Federated Hyperparameter Search

Code and data for the paper *"How Much Should Clients Agree on Hyperparameters? Characterizing
Coupling Regimes in Federated Hyperparameter Search"*.

The study compares hyperparameter-search designs for federated learning that differ in **HP
coupling**, i.e. how correlated the hyperparameters used by different clients within a round are:
per-client GA (zero coupling), per-client GA with a shared surrogate (medium), and server-broadcast
GA (high), plus fixed-HP anchors and non-GA broadcast searchers. Stack: Flower + PyTorch, CIFAR-10
and FEMNIST.

## Quick start

```bash
make install
make smoke                                        # 2 rounds, CPU, ~6 min
make run CONFIG=configs/ga_broadcast_cifar.yaml SEED=0   # a real experiment
```

All scenarios are YAML files; see [`configs/README.md`](configs/README.md) for the full matrix
and schema. Multi-seed sweeps:

```bash
federatedgeneticalgorithm/.venv/bin/python scripts/run_matrix.py \
    --configs configs/ga_broadcast_cifar.yaml --seeds 0 1 2 3 4 \
    --federation local-simulation-gpu --min-eval-rounds 20
```

Results append to `federatedgeneticalgorithm/artifacts/matrix_summary.csv` (resume-safe: completed
pairs are skipped, failed pairs retried). Launch sweeps with the venv interpreter directly, not via
`uv run`: Ray propagates the `uv` wrapper into its workers, which then resolve the wrong project.

## Reproducing the paper

Every table and figure regenerates from the committed telemetry:

```bash
uv run --project federatedgeneticalgorithm python analysis/results_tables.py   # tables
uv run --project federatedgeneticalgorithm python analysis/fitness_bias.py    # bias traces
uv run --project federatedgeneticalgorithm python analysis/paper_figures.py            # figures
```

## Layout

```
configs/                 scenario definitions (see configs/README.md)
scripts/run_matrix.py    (config × seed) sweep orchestrator, resume-safe
federatedgeneticalgorithm/
  federatedgeneticalgorithm/   Flower app: client, server, GA, surrogate, baselines, telemetry
  artifacts/                   matrix_summary.csv + per-run telemetry
analysis/                analysis scripts + generated reports (analysis/README.md)
```
