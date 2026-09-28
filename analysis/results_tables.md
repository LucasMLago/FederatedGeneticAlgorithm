# Results tables

_Fonte: `federatedgeneticalgorithm/artifacts/matrix_summary_final.csv` + telemetria por round em `artifacts/runs/`. Runs ok: 110._


## CIFAR-10 / ResNet 11M / α=0.5 (cenário principal)

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_expert_cifar` | fixed HP (expert) | 5 | 84.47 ± 0.48 | 83.70 ± 0.36 | 50.2 | 0 (0/5) | 0/5 | 5 / 6 / 9 |
| `fixed_naive_cifar` | fixed HP (naive) | 5 | 78.22 ± 0.91 | 77.92 ± 0.93 | 50.4 | 0 (0/5) | 0/5 | 11 / 16 / — |
| `ga_perclient_cifar` | GA zero-coupling | 5 | 83.21 ± 0.66 | 82.77 ± 1.19 | 142.7 | 0 (0/5) | 0/5 | 5 / 9 / 12 |
| `ga_surrogate_cifar` | GA medium-coupling | 5 | 77.91 ± 6.09 | 66.84 ± 28.33 | 63.7 | 7 (3/5) | 0/5 | 8 / 14 / 16 |
| `ga_broadcast_cifar` | GA high-coupling | 5 | 79.07 ± 1.86 | 78.21 ± 2.18 | 51.4 | 4 (3/5) | 0/5 | 10 / 12 / 12 |
| `rs_broadcast_cifar` | RS high-coupling | 5 | 76.14 ± 3.91 | 65.42 ± 20.66 | 52.3 | 10 (5/5) | 0/5 | 14 / 12 / 18 |
| `tpe_broadcast_cifar` | TPE high-coupling | 5 | 78.56 ± 3.52 | 78.11 ± 3.18 | 50.7 | 4 (4/5) | 0/5 | 14 / 16 / 19 |

## FEMNIST / CNN LEAF / α=0.5

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_expert_femnist` | fixed HP (expert) | 3 | 83.07 ± 0.26 | 82.07 ± 0.36 | 9.5 | 0 (0/3) | 0/3 | 3 / 4 / 7 |
| `fixed_naive_femnist` | fixed HP (naive) | 3 | 83.09 ± 0.08 | 82.89 ± 0.26 | 10.3 | 0 (0/3) | 0/3 | 2 / 3 / 7 |
| `ga_perclient_femnist` | GA zero-coupling | 3 | 83.06 ± 0.34 | 82.33 ± 0.60 | 22.4 | 0 (0/3) | 0/3 | 3 / 4 / 7 |
| `ga_surrogate_femnist` | GA medium-coupling | 3 | 82.91 ± 0.42 | 82.86 ± 0.46 | 12.7 | 1 (1/3) | 0/3 | 3 / 4 / 8 |
| `ga_broadcast_femnist` | GA high-coupling | 3 | 83.06 ± 0.16 | 81.86 ± 1.87 | 11.3 | 1 (1/3) | 0/3 | 3 / 4 / 9 |
| `rs_broadcast_femnist` | RS high-coupling | 3 | 82.82 ± 0.73 | 81.63 ± 1.21 | 10.3 | 3 (3/3) | 0/3 | 5 / 7 / 11 |
| `tpe_broadcast_femnist` | TPE high-coupling | 3 | 83.08 ± 0.60 | 82.63 ± 0.46 | 10.6 | 2 (2/3) | 0/3 | 3 / 4 / 8 |

## CIFAR-10 / SmallCNN 530K / α=0.5

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_expert_smallcnn` | fixed HP (expert) | 3 | 66.32 ± 1.44 | 66.32 ± 1.44 | 8.4 | 0 (0/3) | 0/3 | — / — / — |
| `fixed_naive_smallcnn` | fixed HP (naive) | 3 | 61.08 ± 1.70 | 60.81 ± 1.57 | 8.1 | 0 (0/3) | 0/3 | — / — / — |
| `ga_perclient_smallcnn` | GA zero-coupling | 3 | 61.06 ± 1.00 | 59.53 ± 2.63 | 22.5 | 0 (0/3) | 0/3 | — / — / — |
| `ga_surrogate_smallcnn` | GA medium-coupling | 3 | 60.92 ± 2.33 | 59.77 ± 2.96 | 12.5 | 0 (0/3) | 0/3 | — / — / — |
| `ga_broadcast_smallcnn` | GA high-coupling | 3 | 64.01 ± 2.06 | 62.29 ± 0.38 | 9.8 | 0 (0/3) | 0/3 | — / — / — |

## CIFAR-10 / ResNet 11M / α=0.1 (boundary)

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_expert_alpha01` | fixed HP (expert) | 3 | 72.39 ± 2.02 | 67.73 ± 5.91 | 50.3 | 1 (1/3) | 0/3 | 17 / — / — |
| `fixed_naive_alpha01` | fixed HP (naive) | 3 | 54.58 ± 5.84 | 48.97 ± 3.53 | 52.1 | 4 (2/3) | 0/3 | — / — / — |
| `ga_perclient_alpha01` | GA zero-coupling | 3 | 64.05 ± 2.32 | 60.77 ± 1.99 | 136.8 | 1 (1/3) | 0/3 | — / — / — |
| `ga_surrogate_alpha01` | GA medium-coupling | 3 | 59.15 ± 9.22 | 51.59 ± 15.49 | 67.1 | 9 (3/3) | 0/3 | — / — / — |
| `ga_broadcast_alpha01` | GA high-coupling | 3 | 56.83 ± 9.00 | 44.18 ± 7.80 | 51.7 | 6 (3/3) | 0/3 | — / — / — |

## Braços de failure-mode do fitness signal (α=0.5)

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ga_broadcast_randominit_cifar` | broadcast GA, random-init pop | 3 | 80.39 ± 1.15 | 79.74 ± 1.91 | 49.3 | 6 (3/3) | 0/3 | 8 / 9 / 17 |
| `ga_broadcast_deltafitness_cifar` | broadcast GA, delta fitness | 3 | 81.95 ± 1.09 | 79.07 ± 2.37 | 52.6 | 5 (3/3) | 0/3 | 8 / 11 / 16 |

## Horizonte de 40 rounds (CIFAR-10, α=0.5)

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ga_perclient_cifar_r40` | per-client GA, 40 rounds | 3 | 85.54 ± 0.16 | 85.00 ± 0.45 | 252.3 | 0 (0/3) | 0/3 | 5 / 9 / 11 |
| `ga_broadcast_cifar_r40` | broadcast GA, 40 rounds | 5 | 83.67 ± 1.12 | 82.51 ± 1.98 | 101.8 | 2 (2/5) | 0/5 | 8 / 11 / 20 |

## Ablações: pool compartilhado do surrogate e elitismo do broadcast GA

| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ga_surrogate_nopool_cifar` | surrogate GA, no shared pool | 5 | 81.10 ± 1.18 | 79.40 ± 3.16 | 62.4 | 7 (4/5) | 0/5 | 8 / 13 / 19 |
| `ga_broadcast_noelite_cifar` | broadcast GA, no elitism | 5 | 78.79 ± 2.23 | 77.56 ± 4.28 | 51.1 | 2 (2/5) | 0/5 | 9 / 13 / 18 |

## Runs com eventos de robustez (drop severo ou colapso)

| Cenário | Seed | run_id | Drops | Max drop (pp) | Round do pior drop | Rounds parseados/total | Colapso |
|---|---:|---|---:|---:|---:|---:|---:|
| `fixed_expert_alpha01` | 0 | `20260810_172002` | 1 | 12.1 | 17 | 20/20 | não |
| `fixed_naive_alpha01` | 1 | `20260813_212134` | 2 | 15.0 | 18 | 20/20 | não |
| `fixed_naive_alpha01` | 2 | `20260813_221358` | 2 | 10.6 | 9 | 20/20 | não |
| `ga_broadcast_alpha01` | 0 | `20260926_012428` | 2 | 18.6 | 7 | 20/20 | não |
| `ga_broadcast_alpha01` | 1 | `20260926_021059` | 3 | 22.5 | 15 | 20/20 | não |
| `ga_broadcast_alpha01` | 2 | `20260926_030539` | 1 | 11.4 | 13 | 20/20 | não |
| `ga_broadcast_cifar` | 0 | `20260924_210944` | 1 | 12.4 | 6 | 20/20 | não |
| `ga_broadcast_cifar` | 1 | `20260924_220245` | 2 | 27.1 | 12 | 20/20 | não |
| `ga_broadcast_cifar` | 3 | `20260924_234423` | 1 | 65.8 | 15 | 20/20 | não |
| `ga_broadcast_cifar_r40` | 1 | `20260927_022839` | 1 | 34.0 | 3 | 40/40 | não |
| `ga_broadcast_cifar_r40` | 3 | `20260928_040825` | 1 | 59.6 | 15 | 40/40 | não |
| `ga_broadcast_deltafitness_cifar` | 0 | `20260926_221540` | 1 | 10.6 | 7 | 20/20 | não |
| `ga_broadcast_deltafitness_cifar` | 1 | `20260926_230722` | 3 | 46.3 | 12 | 20/20 | não |
| `ga_broadcast_deltafitness_cifar` | 2 | `20260927_000038` | 1 | 16.7 | 4 | 20/20 | não |
| `ga_broadcast_femnist` | 1 | `20260926_041115` | 1 | 24.9 | 3 | 20/20 | não |
| `ga_broadcast_noelite_cifar` | 1 | `20260927_211723` | 1 | 30.9 | 3 | 20/20 | não |
| `ga_broadcast_noelite_cifar` | 3 | `20260927_225851` | 1 | 61.8 | 12 | 20/20 | não |
| `ga_broadcast_randominit_cifar` | 0 | `20260926_194739` | 1 | 10.6 | 6 | 20/20 | não |
| `ga_broadcast_randominit_cifar` | 1 | `20260926_203536` | 2 | 53.7 | 12 | 20/20 | não |
| `ga_broadcast_randominit_cifar` | 2 | `20260926_212554` | 3 | 13.5 | 3 | 20/20 | não |
| `ga_perclient_alpha01` | 0 | `20260926_082023` | 1 | 12.3 | 9 | 20/20 | não |
| `ga_surrogate_alpha01` | 0 | `20260926_162613` | 2 | 14.5 | 17 | 20/20 | não |
| `ga_surrogate_alpha01` | 1 | `20260926_174017` | 4 | 29.9 | 18 | 20/20 | não |
| `ga_surrogate_alpha01` | 2 | `20260926_184516` | 3 | 28.2 | 17 | 20/20 | não |
| `ga_surrogate_cifar` | 1 | `20260925_210814` | 2 | 45.0 | 9 | 20/20 | não |
| `ga_surrogate_cifar` | 3 | `20260925_231543` | 4 | 51.6 | 20 | 20/20 | não |
| `ga_surrogate_cifar` | 4 | `20260926_001816` | 1 | 11.2 | 18 | 20/20 | não |
| `ga_surrogate_femnist` | 1 | `20260926_152329` | 1 | 19.3 | 3 | 20/20 | não |
| `ga_surrogate_nopool_cifar` | 0 | `20260927_151415` | 2 | 13.2 | 11 | 20/20 | não |
| `ga_surrogate_nopool_cifar` | 1 | `20260927_161459` | 1 | 13.0 | 11 | 20/20 | não |
| `ga_surrogate_nopool_cifar` | 2 | `20260927_171943` | 3 | 24.5 | 9 | 20/20 | não |
| `ga_surrogate_nopool_cifar` | 3 | `20260927_182114` | 1 | 17.6 | 7 | 20/20 | não |
| `rs_broadcast_cifar` | 0 | `20260925_012709` | 2 | 43.2 | 4 | 20/20 | não |
| `rs_broadcast_cifar` | 1 | `20260925_022117` | 3 | 40.1 | 20 | 20/20 | não |
| `rs_broadcast_cifar` | 2 | `20260925_031324` | 2 | 31.7 | 5 | 20/20 | não |
| `rs_broadcast_cifar` | 3 | `20260928_004135` | 2 | 25.4 | 4 | 20/20 | não |
| `rs_broadcast_cifar` | 4 | `20260928_013253` | 1 | 45.5 | 7 | 20/20 | não |
| `rs_broadcast_femnist` | 0 | `20260926_050313` | 1 | 42.5 | 4 | 20/20 | não |
| `rs_broadcast_femnist` | 1 | `20260926_051339` | 1 | 25.3 | 3 | 20/20 | não |
| `rs_broadcast_femnist` | 2 | `20260926_052347` | 1 | 13.6 | 5 | 20/20 | não |
| `tpe_broadcast_cifar` | 0 | `20260925_040449` | 1 | 10.6 | 3 | 20/20 | não |
| `tpe_broadcast_cifar` | 1 | `20260925_045415` | 1 | 18.2 | 9 | 20/20 | não |
| `tpe_broadcast_cifar` | 2 | `20260925_054600` | 1 | 23.2 | 9 | 20/20 | não |
| `tpe_broadcast_cifar` | 4 | `20260928_031813` | 1 | 17.1 | 7 | 20/20 | não |
| `tpe_broadcast_femnist` | 0 | `20260926_053411` | 1 | 18.4 | 3 | 20/20 | não |
| `tpe_broadcast_femnist` | 1 | `20260926_054520` | 1 | 14.6 | 9 | 20/20 | não |

## Testes pareados (peak e final)

| Comparação | Δpeak (pp, a−b) | MW-U p | Wilcoxon p (N pareado) | Δfinal (pp) | MW-U p | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR: per-client vs FedGA | +4.14 | 0.008 | 0.062 (N=5) | +4.56 | 0.016 | 0.062 |
| CIFAR: per-client vs surrogate | +5.30 | 0.056 | 0.062 (N=5) | +15.93 | 0.095 | 0.188 |
| CIFAR: surrogate vs FedGA | -1.16 | 0.841 | 1.000 (N=5) | -11.37 | 1.000 | 1.000 |
| CIFAR: FedGA vs RS (broadcast family) | +2.93 | 0.310 | 0.188 (N=5) | +12.79 | 0.032 | 0.125 |
| CIFAR: FedGA vs TPE (broadcast family) | +0.52 | 1.000 | 1.000 (N=5) | +0.10 | 0.841 | 1.000 |
| CIFAR: per-client vs naive baseline | +4.99 | 0.008 | 0.062 (N=5) | +4.84 | 0.008 | 0.062 |
| CIFAR: FedGA vs naive baseline | +0.86 | 0.548 | 0.625 (N=5) | +0.28 | 0.690 | 1.000 |
| CIFAR: expert vs per-client | +1.26 | 0.016 | 0.062 (N=5) | +0.93 | 0.222 | 0.312 |
| Broadcast GA: seeded vs random-init population | -1.32 | 0.250 | 0.500 (N=3) | -1.53 | 0.571 | 0.500 |
| Broadcast GA: absolute vs delta fitness | -2.87 | 0.143 | 0.250 (N=3) | -0.86 | 0.571 | 1.000 |
| FEMNIST: per-client vs FedGA | +0.01 | 1.000 | 1.000 (N=3) | +0.47 | 1.000 | 1.000 |
| FEMNIST: expert vs naive | -0.02 | 1.000 | 1.000 (N=3) | -0.82 | 0.100 | 0.250 |
| FEMNIST: expert vs FedGA | +0.01 | 1.000 | 1.000 (N=3) | +0.22 | 0.700 | 1.000 |
| Small: expert vs per-client GA | +5.26 | 0.100 | 0.250 (N=3) | +6.79 | 0.100 | 0.250 |
| Small: expert vs FedGA | +2.31 | 0.400 | 0.500 (N=3) | +4.03 | 0.100 | 0.250 |
| Small: per-client vs FedGA | -2.96 | 0.100 | 0.250 (N=3) | -2.76 | 0.200 | 0.250 |
| Small: per-client vs surrogate | +0.13 | 1.000 | 1.000 (N=3) | -0.23 | 1.000 | 1.000 |
| α=0.1: expert vs per-client GA | +8.33 | 0.100 | 0.250 (N=3) | +6.95 | 0.200 | 0.500 |
| α=0.1: expert vs FedGA | +15.56 | 0.100 | 0.250 (N=3) | +23.55 | 0.100 | 0.250 |
| α=0.1: per-client vs FedGA | +7.23 | 0.200 | 0.250 (N=3) | +16.60 | 0.100 | 0.250 |
| α=0.1: per-client vs surrogate | +4.91 | 0.400 | 0.500 (N=3) | +9.18 | 0.700 | 0.750 |
| α=0.1: surrogate vs FedGA | +2.32 | 0.700 | 0.750 (N=3) | +7.41 | 0.400 | 0.750 |
| α=0.1: expert vs naive | +17.81 | 0.100 | 0.250 (N=3) | +18.75 | 0.100 | 0.250 |
| α=0.1: naive vs FedGA | -2.25 | 0.700 | 0.750 (N=3) | +4.80 | 0.700 | 0.750 |
| 40 rounds: per-client vs FedGA | +1.86 | 0.036 | 0.250 (N=3) | +2.49 | 0.143 | 0.500 |
| Per-client: 40 vs 20 rounds | +2.33 | 0.036 | 0.250 (N=3) | +2.23 | 0.036 | 0.250 |
| FedGA: 40 vs 20 rounds | +4.60 | 0.008 | 0.062 (N=5) | +4.30 | 0.016 | 0.062 |
| FedGA 40 rounds vs per-client 20 rounds | +0.46 | 0.548 | 0.625 (N=5) | -0.26 | 1.000 | 0.812 |
| Surrogate: no shared pool vs shared pool | +3.19 | 0.841 | 0.438 (N=5) | +12.57 | 0.841 | 0.625 |
| CIFAR: per-client vs surrogate without pool | +2.11 | 0.008 | 0.062 (N=5) | +3.36 | 0.056 | 0.125 |
| FedGA: no elitism vs elitism | -0.28 | 0.841 | 1.000 (N=5) | -0.65 | 0.841 | 1.000 |
