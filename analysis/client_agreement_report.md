# Client agreement and harmful configurations

_Fonte: `federatedgeneticalgorithm/artifacts/matrix_summary_final.csv` + `client_round_metrics.csv` de cada run. Nocivas: lion com lr >= 5e-3, adam/adamw/radam com lr = 1e-2._


## Regimes em que cada cliente escolhe

| Cenário | Runs | Pares idênticos (round >= 5) | Pares com mesmo otimizador | Atualizações nocivas | Quedas com cliente nocivo | Rounds sem queda com cliente nocivo |
|---|---:|---:|---:|---:|---:|---:|
| `ga_perclient_cifar` | 10 | 1% | 29% | 1.2% | 0/1 | 6% |
| `ga_surrogate_cifar` | 10 | 3% | 46% | 5.9% | 8/9 | 22% |
| `ga_surrogate_nopool_cifar` | 5 | 1% | 33% | 6.2% | 7/7 | 23% |
| `ga_perclient_cifar_r40` | 3 | 1% | 45% | 1.2% | 0/0 | 4% |
| `ga_perclient_alpha01` | 3 | 1% | 32% | 0.7% | 0/1 | 4% |
| `ga_surrogate_alpha01` | 3 | 1% | 56% | 12.3% | 6/9 | 33% |
| `fedex_cifar` | 10 | 60% | 87% | 2.8% | 1/2 | 6% |
| `ga_surrogate_longeval_cifar` | 5 | 2% | 59% | 1.0% | 0/0 | 5% |

## Busca por broadcast (uma configuração por round para todos)

| Cenário | Runs | Rounds com configuração nociva | Quedas | Quedas em round nocivo | Rounds até voltar a 5 pp do valor anterior |
|---|---:|---:|---:|---:|---|
| `ga_broadcast_cifar` | 10 | 24/200 | 11 | 8/11 | 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, — |
| `ga_broadcast_noelite_cifar` | 5 | 11/100 | 2 | 2/2 | 2, 1 |
| `ga_broadcast_cifar_r40` | 5 | 15/200 | 2 | 2/2 | 1, 1 |
| `tpe_broadcast_cifar` | 10 | 32/200 | 11 | 8/11 | 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 |
| `rs_broadcast_cifar` | 10 | 54/200 | 17 | 16/17 | 2, 2, 1, 1, —, 1, 1, 1, 1, 1, 1, 1, 1, —, 3, 1, 1 |

## Conjunto nocivo dentro e fora da amostra (seeds 0–4 vs 5–9)

| Cenário | Seeds | Rounds com configuração nociva | Quedas | Quedas em round nocivo |
|---|---|---:|---:|---:|
| `ga_broadcast_cifar` | 0–4 | 13/100 | 4 | 4/4 |
| `ga_broadcast_cifar` | 5–9 | 11/100 | 7 | 4/7 |
| `tpe_broadcast_cifar` | 0–4 | 16/100 | 4 | 3/4 |
| `tpe_broadcast_cifar` | 5–9 | 16/100 | 7 | 5/7 |
| `rs_broadcast_cifar` | 0–4 | 28/100 | 10 | 10/10 |
| `rs_broadcast_cifar` | 5–9 | 26/100 | 7 | 6/7 |

## FedEx: concentração da distribuição do servidor

| Seed | Prob. da HP mais provável no round 10 | No round 20 | HP mais provável no round 20 |
|---:|---:|---:|---|
| 0 | 0.39 | 0.98 | lion, lr 0.0005, batch 128 |
| 1 | 0.60 | 1.00 | sgd, lr 0.003, batch 128 |
| 2 | 0.84 | 1.00 | radam, lr 0.001, batch 64 |
| 3 | 0.97 | 0.56 | radam, lr 0.0005, batch 128 |
| 4 | 0.52 | 0.93 | sgd, lr 0.005, batch 128 |
| 5 | 0.32 | 1.00 | adamw, lr 0.001, batch 128 |
| 6 | 0.97 | 1.00 | radam, lr 0.003, batch 128 |
| 7 | 0.69 | 0.99 | lion, lr 0.0005, batch 64 |
| 8 | 0.25 | 0.90 | sgd, lr 0.003, batch 64 |
| 9 | 0.49 | 1.00 | sgd, lr 0.003, batch 128 |
