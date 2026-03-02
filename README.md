# Federated Genetic Algorithm

Federated Learning simulation using [Flower](https://flower.ai/) where each client runs a **Genetic Algorithm** to optimize hyperparameters locally before training a CNN on CIFAR-10 / MNIST.

## Project structure

```
.
├── data/                        # CIFAR-10 and MNIST datasets (local)
├── federatedgeneticalgorithm/   # Flower app (server + clients)
│   ├── federatedgeneticalgorithm/
│   │   ├── client_app.py        # ClientApp — GA + local training
│   │   ├── server_app.py        # ServerApp — aggregation strategy
│   │   ├── genetic_algorithm.py # DEAP-based GA for HP search
│   │   ├── surrogate_model.py   # RandomForest surrogate for fitness estimation
│   │   ├── task.py              # Model definition, train/test helpers
│   │   └── config/config.py     # Global config (device, seed, …)
│   ├── scripts/
│   │   ├── run_federated_learning_cpu.sh
│   │   └── run_federated_learning_gpu.sh
│   └── pyproject.toml           # Flower app config & dependencies
├── notebooks/                   # Exploratory notebooks
└── pyproject.toml               # Root workspace dependencies
```

## Requirements

- Python 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) **or** pip

## Installation

```bash
# 1. Create and activate a virtual environment
uv venv --python 3.10
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# 2. Install the Flower app and its dependencies
cd federatedgeneticalgorithm
pip install -e .
```

## Running

All commands below must be run from inside the `federatedgeneticalgorithm/` directory.

### CPU (default)

```bash
flwr run .
```

### GPU

```bash
flwr run . local-simulation-gpu
```

Alternatively, use the provided scripts from the project root:

```bash
bash federatedgeneticalgorithm/scripts/run_federated_learning_cpu.sh
bash federatedgeneticalgorithm/scripts/run_federated_learning_gpu.sh
```

### Tuning the run

Key hyperparameters live in `federatedgeneticalgorithm/pyproject.toml` under `[tool.flwr.app.config]`:

| Key                 | Default | Description                        |
|---------------------|---------|------------------------------------|
| `num-server-rounds` | `20`    | Number of federated rounds         |
| `fraction-train`    | `0.5`   | Fraction of clients selected/round |
| `local-epochs`      | `5`     | Local training epochs per client   |

Override any value at runtime without editing the file:

```bash
flwr run . --run-config "num-server-rounds=10 local-epochs=3"
```

