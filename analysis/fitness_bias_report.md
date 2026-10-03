# Fitness-signal bias report

_Runs reconstructed from training.log: 16 (ga_broadcast_cifar, ga_broadcast_deltafitness_cifar, ga_broadcast_randominit_cifar)_


## 1. Cold-start bias: rank of the first-evaluated individual (gen 0)

| Arm | Seed | Fitness R1 | Fitness R2-R4 (gen 0) | Rank of 1st (1=best, 4=worst) | 1st HP re-broadcast after gen 0? |
|---|---:|---:|---|---:|---|
| seeded population | 0 | 0.1143 | 0.209, 0.562, 0.664 | 4 | NO |
| seeded population | 1 | 0.1496 | 0.369, 0.131, 0.422 | 3 | NO |
| seeded population | 2 | 0.1205 | 0.536, 0.549, 0.503 | 4 | NO |
| seeded population | 3 | 0.1533 | 0.335, 0.580, 0.625 | 4 | NO |
| seeded population | 4 | 0.1705 | 0.565, 0.614, 0.607 | 4 | NO |
| seeded population | 5 | 0.1301 | 0.308, 0.518, 0.527 | 4 | NO |
| seeded population | 6 | 0.1592 | 0.147, 0.507, 0.519 | 3 | NO |
| seeded population | 7 | 0.1824 | 0.243, 0.382, 0.386 | 4 | NO |
| seeded population | 8 | 0.1569 | 0.429, 0.512, 0.412 | 4 | NO |
| seeded population | 9 | 0.2037 | 0.579, 0.461, 0.640 | 4 | NO |
| random-init population | 0 | 0.1034 | 0.463, 0.545, 0.676 | 4 | NO |
| random-init population | 1 | 0.2466 | 0.103, 0.323, 0.283 | 3 | NO |
| random-init population | 2 | 0.1520 | 0.275, 0.139, 0.622 | 3 | NO |

**9/13 runs** rank the first-evaluated individual worst of generation 0 (no-bias base rate: 25%; expected mean rank without bias: 2.5).


## 2. Trajectory-position bias: post-crash delta credit (delta-fitness arm)

| Seed | Crash (>10pp) | Post-crash round: HP | Credited delta | Promoted to best-so-far? | Run peak/final |
|---:|---|---|---:|---|---|
| 0 | R7 (−11pp) | radam/lr=0.005/wd=0.0001/mom=0.7/b=128 | +0.103 | no | 83.2 / 79.7 |
| 1 | R3 (−29pp) | lion/lr=0.0005/wd=0.001/mom=0.95/b=64 | +0.290 | **YES** | 81.2 / 81.1 |
| 1 | R7 (−10pp) | lion/lr=0.0005/wd=0.001/mom=0.95/b=64 | +0.117 | no | 81.2 / 81.1 |
| 1 | R12 (−46pp) | lion/lr=0.0005/wd=0.001/mom=0.95/b=64 | +0.492 | **YES** | 81.2 / 81.1 |
| 2 | R4 (−17pp) | adam/lr=0.0005/wd=0.0001/mom=0.7/b=64 | +0.260 | no | 81.4 / 76.5 |

### Stale elite (best-delta staleness)

| Seed | Best-delta HP | Earned at | Delta | Later re-broadcasts | Mean delta on re-evaluation |
|---:|---|---|---:|---:|---:|
| 0 | radam/lr=0.005/wd=0.0001/mom=0.7/b=128 | R3 | +0.256 | 8 | +0.016 |
| 1 | lion/lr=0.0005/wd=0.001/mom=0.95/b=64 | R13 | +0.492 | 2 | +0.022 |
| 2 | adam/lr=0.0005/wd=0.0001/mom=0.7/b=64 | R2 | +0.283 | 6 | +0.054 |

## 3. Peak accuracy per arm (context)

- `ga_broadcast_cifar` (seeded population): peak 78.64 ± 2.67 (N=10)
- `ga_broadcast_deltafitness_cifar` (delta fitness): peak 81.95 ± 1.09 (N=3)
- `ga_broadcast_randominit_cifar` (random-init population): peak 80.39 ± 1.15 (N=3)

> Note: seeding has no measurable effect on end-point accuracy (arms tie); the cold-start bias wastes the warm start (the expert HP is discarded as generation-worst) rather than degrading the mean outcome.
