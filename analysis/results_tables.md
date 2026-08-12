# Results tables

_Fonte: `federatedgeneticalgorithm/artifacts/matrix_summary.csv` + telemetria por round em `artifacts/runs/`. Runs ok: 88._


## CIFAR-10 / ResNet 11M / α=0.5 (cenário principal)

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_expert_cifar` | fixed HP (expert) | 5 | 84.47 ± 0.48 | 83.70 ± 0.36 | 50.2 | 0 (0/5) | 0/5 | 5 / 6 / 9 |
| `fixed_naive_cifar` | fixed HP (naive) | 5 | 78.22 ± 0.91 | 77.92 ± 0.93 | 50.4 | 0 (0/5) | 0/5 | 11 / 16 / — |
| `ga_perclient_cifar` | GA zero-coupling | 5 | 81.94 ± 0.68 | 81.19 ± 1.62 | 114.2 | 0 (0/5) | 0/5 | 7 / 9 / 14 |
| `ga_surrogate_cifar` | GA medium-coupling | 5 | 80.69 ± 1.96 | 79.50 ± 3.08 | 63.8 | 4 (2/5) | 0/5 | 11 / 15 / 17 |
| `ga_broadcast_cifar` | GA high-coupling | 5 | 81.07 ± 1.89 | 79.75 ± 2.82 | 51.2 | 4 (3/5) | 0/5 | 10 / 13 / 16 |
| `rs_broadcast_cifar` | RS high-coupling | 3 | 73.96 ± 5.92 | 58.75 ± 24.91 | 57.9 | 8 (3/3) | 0/3 | 12 / 16 / — |
| `tpe_broadcast_cifar` | TPE high-coupling | 3 | 78.14 ± 5.61 | 77.09 ± 6.83 | 51.5 | 2 (2/3) | 0/3 | 11 / 12 / 16 |

## FEMNIST / CNN LEAF / α=0.5

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_expert_femnist` | fixed HP (expert) | 3 | 83.07 ± 0.26 | 82.07 ± 0.36 | 9.5 | 0 (0/3) | 0/3 | 3 / 4 / 7 |
| `fixed_naive_femnist` | fixed HP (naive) | 3 | 83.09 ± 0.08 | 82.89 ± 0.26 | 10.3 | 0 (0/3) | 0/3 | 2 / 3 / 7 |
| `ga_perclient_femnist` | GA zero-coupling | 3 | 82.65 ± 0.29 | 81.47 ± 1.03 | 17.8 | 0 (0/3) | 0/3 | 3 / 4 / 8 |
| `ga_surrogate_femnist` | GA medium-coupling | 3 | 83.08 ± 0.40 | 82.73 ± 0.58 | 14.4 | 0 (0/3) | 0/3 | 2 / 4 / 8 |
| `ga_broadcast_femnist` | GA high-coupling | 3 | 82.98 ± 0.50 | 82.29 ± 0.42 | 9.6 | 1 (1/3) | 0/3 | 3 / 5 / 9 |
| `rs_broadcast_femnist` | RS high-coupling | 3 | 82.80 ± 0.58 | 81.62 ± 0.37 | 9.3 | 5 (3/3) | 0/3 | 6 / 7 / 10 |
| `tpe_broadcast_femnist` | TPE high-coupling | 3 | 82.84 ± 0.44 | 82.15 ± 0.80 | 9.4 | 0 (0/3) | 0/3 | 3 / 4 / 8 |

## CIFAR-10 / SmallCNN 530K / α=0.5

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_expert_smallcnn` | fixed HP (expert) | 3 | 66.32 ± 1.44 | 66.32 ± 1.44 | 8.4 | 0 (0/3) | 0/3 | — / — / — |
| `fixed_naive_smallcnn` | fixed HP (naive) | 3 | 61.08 ± 1.70 | 60.81 ± 1.57 | 8.1 | 0 (0/3) | 0/3 | — / — / — |
| `ga_perclient_smallcnn` | GA zero-coupling | 3 | 61.96 ± 1.05 | 60.64 ± 1.68 | 16.9 | 0 (0/3) | 0/3 | — / — / — |
| `ga_surrogate_smallcnn` | GA medium-coupling | 3 | 61.07 ± 2.13 | 60.11 ± 3.73 | 12.2 | 3 (1/3) | 0/3 | — / — / — |
| `ga_broadcast_smallcnn` | GA high-coupling | 3 | 62.24 ± 1.66 | 60.91 ± 1.58 | 8.4 | 1 (1/3) | 0/3 | — / — / — |

## CIFAR-10 / ResNet 11M / α=0.1 (boundary)

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_expert_alpha01` | fixed HP (expert) | 3 | 72.39 ± 2.02 | 67.73 ± 5.91 | 50.3 | 1 (1/3) | 0/3 | 17 / — / — |
| `fixed_naive_alpha01` | fixed HP (naive) | 3 | 54.58 ± 5.84 | 48.97 ± 3.53 | 52.1 | 4 (2/3) | 0/3 | — / — / — |
| `ga_perclient_alpha01` | GA zero-coupling | 3 | 65.68 ± 3.01 | 59.33 ± 3.84 | 114.9 | 4 (2/3) | 0/3 | — / — / — |
| `ga_surrogate_alpha01` | GA medium-coupling | 3 | 51.13 ± 10.20 | 39.86 ± 12.18 | 66.7 | 5 (3/3) | 0/3 | — / — / — |
| `ga_broadcast_alpha01` | GA high-coupling | 3 | 59.74 ± 4.14 | 44.54 ± 3.63 | 52.7 | 10 (3/3) | 0/3 | — / — / — |

## Braços de failure-mode do fitness signal (α=0.5)

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ga_broadcast_randominit_cifar` | broadcast GA, random-init pop | 3 | 80.90 ± 1.14 | 79.76 ± 1.17 | 51.3 | 4 (2/3) | 0/3 | 6 / 10 / 17 |
| `ga_broadcast_deltafitness_cifar` | broadcast GA, delta fitness | 3 | 81.16 ± 1.58 | 75.63 ± 5.45 | 58.1 | 3 (2/3) | 0/3 | 7 / 10 / 17 |

## Runs com eventos de robustez (drop severo ou colapso)

| Cenário | Seed | run_id | Drops | Max drop (pp) | Round do pior drop | Rounds parseados/total | Colapso |
|---|---:|---|---:|---:|---:|---:|---:|
| `fixed_expert_alpha01` | 0 | `20260810_172002` | 1 | 12.1 | 17 | 20/20 | não |
| `fixed_naive_alpha01` | 1 | `20260813_212134` | 2 | 15.0 | 18 | 20/20 | não |
| `fixed_naive_alpha01` | 2 | `20260813_221358` | 2 | 10.6 | 9 | 20/20 | não |
| `ga_broadcast_alpha01` | 0 | `20260810_195132` | 3 | 13.4 | 14 | 20/20 | não |
| `ga_broadcast_alpha01` | 1 | `20260810_204833` | 3 | 30.4 | 15 | 20/20 | não |
| `ga_broadcast_alpha01` | 2 | `20260810_213603` | 4 | 18.2 | 4 | 20/20 | não |
| `ga_broadcast_cifar` | 0 | `20260518_214131` | 1 | 11.4 | 6 | 20/20 | não |
| `ga_broadcast_cifar` | 1 | `20260518_223402` | 2 | 32.7 | 12 | 20/20 | não |
| `ga_broadcast_cifar` | 3 | `20260519_001906` | 1 | 59.4 | 15 | 20/20 | não |
| `ga_broadcast_deltafitness_cifar` | 0 | `20260810_104352` | 1 | 12.3 | 20 | 20/20 | não |
| `ga_broadcast_deltafitness_cifar` | 1 | `20260810_124128` | 2 | 29.0 | 3 | 20/20 | não |
| `ga_broadcast_femnist` | 1 | `20260520_160737` | 1 | 27.0 | 3 | 20/20 | não |
| `ga_broadcast_randominit_cifar` | 1 | `20260810_022903` | 2 | 53.1 | 12 | 20/20 | não |
| `ga_broadcast_randominit_cifar` | 2 | `20260810_032034` | 2 | 25.1 | 3 | 20/20 | não |
| `ga_broadcast_smallcnn` | 1 | `20260527_185605` | 1 | 18.4 | 12 | 20/20 | não |
| `ga_perclient_alpha01` | 1 | `20260811_143434` | 2 | 11.9 | 18 | 20/20 | não |
| `ga_perclient_alpha01` | 2 | `20260811_162956` | 2 | 22.8 | 10 | 20/20 | não |
| `ga_surrogate_alpha01` | 0 | `20260813_230715` | 3 | 22.4 | 19 | 20/20 | não |
| `ga_surrogate_alpha01` | 1 | `20260814_001340` | 1 | 16.7 | 16 | 20/20 | não |
| `ga_surrogate_alpha01` | 2 | `20260814_012256` | 1 | 16.3 | 13 | 20/20 | não |
| `ga_surrogate_cifar` | 2 | `20260519_233207` | 2 | 40.3 | 8 | 20/20 | não |
| `ga_surrogate_cifar` | 3 | `20260520_003529` | 2 | 15.8 | 4 | 20/20 | não |
| `ga_surrogate_smallcnn` | 1 | `20260813_200716` | 3 | 26.3 | 8 | 20/20 | não |
| `rs_broadcast_cifar` | 0 | `20260520_184213` | 3 | 38.9 | 4 | 20/20 | não |
| `rs_broadcast_cifar` | 1 | `20260810_143751` | 3 | 39.1 | 15 | 20/20 | não |
| `rs_broadcast_cifar` | 2 | `20260810_153820` | 2 | 32.0 | 11 | 20/20 | não |
| `rs_broadcast_femnist` | 0 | `20260520_232738` | 1 | 40.0 | 4 | 20/20 | não |
| `rs_broadcast_femnist` | 1 | `20260520_233702` | 2 | 17.6 | 3 | 20/20 | não |
| `rs_broadcast_femnist` | 2 | `20260520_234611` | 2 | 24.0 | 5 | 20/20 | não |
| `tpe_broadcast_cifar` | 1 | `20260520_214546` | 1 | 16.4 | 9 | 20/20 | não |
| `tpe_broadcast_cifar` | 2 | `20260520_223553` | 1 | 39.3 | 9 | 20/20 | não |

## Testes pareados (peak e final)

| Comparação | Δpeak (pp, a−b) | MW-U p | Wilcoxon p (N pareado) | Δfinal (pp) | MW-U p | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR: per-client vs FedGA | +0.87 | 1.000 | 0.438 (N=5) | +1.44 | 0.548 | 0.312 |
| CIFAR: per-client vs surrogate | +1.25 | 0.222 | 0.438 (N=5) | +1.69 | 0.421 | 0.188 |
| CIFAR: surrogate vs FedGA | -0.39 | 0.841 | 0.812 (N=5) | -0.25 | 0.841 | 1.000 |
| CIFAR: FedGA vs RS (broadcast family) | +7.12 | 0.036 | 0.250 (N=3) | +21.01 | 0.071 | 0.250 |
| CIFAR: FedGA vs TPE (broadcast family) | +2.93 | 0.393 | 0.250 (N=3) | +2.67 | 0.786 | 0.750 |
| CIFAR: per-client vs naive baseline | +3.72 | 0.008 | 0.062 (N=5) | +3.26 | 0.016 | 0.062 |
| CIFAR: FedGA vs naive baseline | +2.86 | 0.056 | 0.125 (N=5) | +1.83 | 0.310 | 0.188 |
| CIFAR: expert vs per-client | +2.53 | 0.008 | 0.062 (N=5) | +2.51 | 0.008 | 0.062 |
| Broadcast GA: seeded vs random-init population | +0.17 | 0.786 | 0.750 (N=3) | -0.00 | 1.000 | 0.750 |
| Broadcast GA: absolute vs delta fitness | -0.09 | 0.786 | 1.000 (N=3) | +4.12 | 0.393 | 0.500 |
| FEMNIST: per-client vs FedGA | -0.33 | 0.400 | 0.250 (N=3) | -0.82 | 0.400 | 0.500 |
| FEMNIST: expert vs naive | -0.02 | 1.000 | 1.000 (N=3) | -0.82 | 0.100 | 0.250 |
| FEMNIST: expert vs FedGA | +0.09 | 1.000 | 0.750 (N=3) | -0.22 | 0.400 | 0.750 |
| Small: expert vs per-client GA | +4.36 | 0.100 | 0.250 (N=3) | +5.68 | 0.100 | 0.250 |
| Small: expert vs FedGA | +4.08 | 0.100 | 0.250 (N=3) | +5.41 | 0.100 | 0.250 |
| Small: per-client vs FedGA | -0.28 | 1.000 | 0.750 (N=3) | -0.27 | 1.000 | 0.500 |
| Small: per-client vs surrogate | +0.89 | 1.000 | 0.500 (N=3) | +0.53 | 1.000 | 0.750 |
| α=0.1: expert vs per-client GA | +6.70 | 0.100 | 0.250 (N=3) | +8.40 | 0.200 | 0.250 |
| α=0.1: expert vs FedGA | +12.65 | 0.100 | 0.250 (N=3) | +23.19 | 0.100 | 0.250 |
| α=0.1: per-client vs FedGA | +5.94 | 0.200 | 0.250 (N=3) | +14.79 | 0.100 | 0.250 |
| α=0.1: per-client vs surrogate | +14.55 | 0.100 | 0.250 (N=3) | +19.47 | 0.100 | 0.250 |
| α=0.1: surrogate vs FedGA | -8.61 | 0.400 | 0.250 (N=3) | -4.68 | 0.700 | 0.750 |
| α=0.1: expert vs naive | +17.81 | 0.100 | 0.250 (N=3) | +18.75 | 0.100 | 0.250 |
| α=0.1: naive vs FedGA | -5.16 | 0.400 | 0.750 (N=3) | +4.43 | 0.200 | 0.250 |
