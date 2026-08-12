# Experiment scenarios

Each YAML file here defines one experiment arm as a set of overrides applied to
[`config.py`](../federatedgeneticalgorithm/federatedgeneticalgorithm/config/config.py).
Run any of them with:

```bash
make run CONFIG=configs/<file>.yaml SEED=0
```

## Scenario matrix

Main comparison (CIFAR-10, ResNet-11M, Dirichlet α=0.5, 10 clients, 20 rounds):

| Config | Design | HP coupling |
|---|---|---|
| `fixed_expert_cifar` | fixed HPs: sgd, lr=0.01, momentum 0.9 | — (upper anchor) |
| `fixed_naive_cifar` | fixed HPs: adam, lr=1e-3 | — (lower anchor) |
| `ga_perclient_cifar` | each client runs its own GA | zero |
| `ga_surrogate_cifar` | per-client GA + shared surrogate filter | medium |
| `ga_broadcast_cifar` | server GA broadcasts one HP per round | high |
| `rs_broadcast_cifar` | random sampling under the broadcast protocol | high (no memory) |
| `tpe_broadcast_cifar` | TPE under the broadcast protocol | high (density model) |

Fitness-signal ablations (server-broadcast GA):

| Config | Variation |
|---|---|
| `ga_broadcast_randominit_cifar` | random initial population (no expert seeding); cold-start-bias control |
| `ga_broadcast_deltafitness_cifar` | fitness = per-round improvement instead of absolute accuracy |

Boundary scenarios (same designs, different regime):

| Suffix | What changes |
|---|---|
| `*_femnist` | FEMNIST + LEAF-spec CNN (second dataset) |
| `*_smallcnn` | 530K-parameter CNN (model-size ablation) |
| `*_alpha01` | Dirichlet α=0.1 (extreme heterogeneity) |
| `smoke_*` | 2-round smoke tests (~6 min on CPU) |

## Schema

```yaml
name: "<scenario-id>"      # archived into run_metadata.json
description: "<one-line>"  # for humans; ignored by the runner
overrides:                 # UPPER_CASE constants from config.py -> values
  SEED: 0
  NUM_SERVER_ROUNDS: 20
  ENABLE_FED_GA: true
```

The runner validates every key against `config.py` (refuses on typos), writes a resolved JSON
snapshot, and exports it via `FGA_CONFIG_PATH`; Ray workers re-import the snapshot, so server and
clients agree on all values without flag-by-flag propagation.
