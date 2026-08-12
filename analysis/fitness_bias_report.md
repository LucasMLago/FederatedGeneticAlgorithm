# Fitness-signal bias report

_Runs reconstructed from training.log: 11 (ga_broadcast_cifar, ga_broadcast_deltafitness_cifar, ga_broadcast_randominit_cifar)_


## 1. Cold-start bias: rank of the first-evaluated individual (gen 0)

| Arm | Seed | Fitness R1 | Fitness R2-R4 (gen 0) | Rank of 1st (1=best, 4=worst) | 1st HP re-broadcast after gen 0? |
|---|---:|---:|---|---:|---|
| seeded population | 0 | 0.1064 | 0.203, 0.521, 0.649 | 4 | NO |
| seeded population | 1 | 0.2366 | 0.371, 0.156, 0.514 | 3 | NO |
| seeded population | 2 | 0.1148 | 0.460, 0.551, 0.563 | 4 | NO |
| seeded population | 3 | 0.1418 | 0.348, 0.433, 0.577 | 4 | NO |
| seeded population | 4 | 0.1884 | 0.523, 0.598, 0.659 | 4 | NO |
| random-init population | 0 | 0.1324 | 0.282, 0.570, 0.673 | 4 | NO |
| random-init population | 1 | 0.2220 | 0.102, 0.346, 0.305 | 3 | NO |
| random-init population | 2 | 0.1023 | 0.440, 0.189, 0.607 | 4 | NO |

**6/8 runs** rank the first-evaluated individual worst of generation 0 (no-bias base rate: 25%; expected mean rank without bias: 2.5).


## 2. Trajectory-position bias: post-crash delta credit (delta-fitness arm)

| Seed | Crash (>10pp) | Post-crash round: HP | Credited delta | Promoted to best-so-far? | Run peak/final |
|---:|---|---|---:|---|---|
| 0 | R20 (−12pp) | — (crash on final round; no next round) | — | — | 83.0 / 69.5 |
| 1 | R3 (−29pp) | lion/lr=0.0005/wd=0.001/mom=0.95/b=64 | +0.311 | **YES** | 80.4 / 79.8 |
| 1 | R15 (−14pp) | sgd/lr=0.0005/wd=0.001/mom=0.7/b=64 | +0.073 | no | 80.4 / 79.8 |
| 2 | 0 | — | — | — | 80.1 / 77.7 |

### Stale elite (best-delta staleness)

| Seed | Best-delta HP | Earned at | Delta | Later re-broadcasts | Mean delta on re-evaluation |
|---:|---|---|---:|---:|---:|
| 0 | radam/lr=0.005/wd=0.0001/mom=0.7/b=128 | R3 | +0.232 | 5 | +0.002 |
| 1 | lion/lr=0.0005/wd=0.001/mom=0.95/b=64 | R4 | +0.311 | 7 | +0.036 |
| 2 | adam/lr=0.0005/wd=0.0001/mom=0.7/b=64 | R2 | +0.286 | 7 | +0.059 |

## 3. Peak accuracy per arm (context)

- `ga_broadcast_cifar` (seeded population): peak 81.07 ± 1.89 (N=5)
- `ga_broadcast_deltafitness_cifar` (delta fitness): peak 81.16 ± 1.58 (N=3)
- `ga_broadcast_randominit_cifar` (random-init population): peak 80.90 ± 1.14 (N=3)

> Note: seeding has no measurable effect on end-point accuracy (arms tie); the cold-start bias wastes the warm start (the expert HP is discarded as generation-worst) rather than degrading the mean outcome.
