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
