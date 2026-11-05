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
| `fedex_cifar` | FedEx: each client draws its HP from a server distribution updated by exponentiated gradient | from low to high as the distribution concentrates |
| `fedpop_cifar` | FedPop: five trainings side by side, client HPs evolving inside each, the worst trainings copying the best | low inside a training, then converging |

Fitness variants of the broadcast GA:

| Config | Change |
|---|---|
| `ga_broadcast_randominit_cifar` | random initial population, no expert seed |
| `ga_broadcast_deltafitness_cifar` | fitness = accuracy gain over the previous round |

Ablations:

| Config | Change |
|---|---|
| `ga_surrogate_nopool_cifar` | each client's surrogate fits only its own evaluations (no shared pool) |
| `ga_surrogate_longeval_cifar` | both surrogate rungs evaluate like the per-client GA (10 epochs, whole partition) |
| `ga_broadcast_noelite_cifar` | broadcast GA without elitism |

Pruned search space (`FED_EXCLUDE_HARMFUL: true`): the server-side searchers never train a
configuration of the harmful set (lion with lr >= 5e-3; adam, adamw or radam with lr >= 1e-2,
40 of the 320 behaviorally distinct configurations). The GA and random search redraw a proposal
that falls in it; TPE draws the learning rate from the rates each optimizer has left.

| Config | Change |
|---|---|
| `ga_broadcast_pruned_cifar` | broadcast GA on the pruned space |
| `rs_broadcast_pruned_cifar` | random search on the pruned space |
| `tpe_broadcast_pruned_cifar` | TPE on the pruned space |

Probes behind two reviewer questions, one seed each unless noted:

| Config | Change |
|---|---|
| `probes/probe_sgd_*` | fixed SGD variants of the expert (one or two values changed), to see whether anything nearby beats it; written by `scripts/make_probe_configs.py configs/probes` |
| `ga_perclient_lambda0_cifar` | per-client GA without the drift penalty (`FITNESS_DRIFT_PENALTY_LAMBDA` and its warmup start set to 0), seeds 0-2 |

Other scenarios, same designs:

| Suffix | Change |
|---|---|
| `*_femnist` | FEMNIST with the LEAF CNN |
| `*_smallcnn` | ~530K-param CNN on CIFAR-10 |
| `*_alpha01` | Dirichlet alpha=0.1 |
| `smoke_*` | 2-round smoke tests (~6 min on CPU); `smoke_pruned_cifar` runs 3 rounds of the pruned broadcast GA |
| `*_r40` | 40 rounds instead of 20 (per-client GA, broadcast GA and the expert) |

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

The broadcast, random search and TPE configs set `FED_FITNESS_SPLIT: val`: the server scores
candidates on the 20% of training data each client holds out, and the test split is only used
for reporting. The per-client and surrogate configs set `GA_EVOLVE_POPULATION: true`, so offspring
and surrogate proposals can replace the population (the older runs kept it frozen).
