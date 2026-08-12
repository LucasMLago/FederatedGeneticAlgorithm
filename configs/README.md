# Scenarios

Each YAML here is one experiment arm, a set of overrides on top of
[config.py](../federatedgeneticalgorithm/federatedgeneticalgorithm/config/config.py). To run one:

```bash
make run CONFIG=configs/<file>.yaml SEED=0
```

## Main comparison

CIFAR-10, ResNet (~11M params), Dirichlet alpha=0.5, 10 clients, 20 rounds.

| Config | Design | Coupling |
|---|---|---|
| `fixed_expert_cifar` | fixed HPs: sgd, lr 0.01, momentum 0.9 | none (expert baseline) |
| `fixed_naive_cifar` | fixed HPs: adam, lr 1e-3 | none (naive baseline) |
| `ga_perclient_cifar` | each client runs its own GA | zero |
| `ga_surrogate_cifar` | per-client GA + shared surrogate filter | medium |
| `ga_broadcast_cifar` | server GA, one HP per round for everyone | high |
| `rs_broadcast_cifar` | random search, broadcast protocol | high, no memory |
| `tpe_broadcast_cifar` | TPE, broadcast protocol | high |

Fitness variants of the broadcast GA:

| Config | Change |
|---|---|
| `ga_broadcast_randominit_cifar` | random initial population, no expert seed |
| `ga_broadcast_deltafitness_cifar` | fitness = accuracy gain over the previous round |

Other scenarios, same designs:

| Suffix | Change |
|---|---|
| `*_femnist` | FEMNIST with the LEAF CNN |
| `*_smallcnn` | ~530K-param CNN on CIFAR-10 |
| `*_alpha01` | Dirichlet alpha=0.1 |
| `smoke_*` | 2-round smoke tests (~6 min on CPU) |

## Format

```yaml
name: "<scenario-id>"      # saved in run_metadata.json
description: "<one line>"  # only for people, the runner ignores it
overrides:                 # UPPER_CASE names from config.py
  SEED: 0
  NUM_SERVER_ROUNDS: 20
  ENABLE_FED_GA: true
```

The runner rejects keys that don't exist in config.py, writes the resolved values to a JSON file
and points `FGA_CONFIG_PATH` at it. Ray workers read the same file, so the server and the clients
always see the same config.
