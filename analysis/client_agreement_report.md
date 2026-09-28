# Client agreement and harmful configurations

_Fonte: `federatedgeneticalgorithm/artifacts/matrix_summary_final.csv` + `client_round_metrics.csv` de cada run. Nocivas: lion com lr >= 5e-3, adam/adamw/radam com lr = 1e-2._


## Regimes em que cada cliente escolhe

| Cenário | Runs | Pares idênticos (round >= 5) | Pares com mesmo otimizador | Atualizações nocivas | Quedas com cliente nocivo | Rounds sem queda com cliente nocivo |
|---|---:|---:|---:|---:|---:|---:|
| `ga_perclient_cifar` | 5 | 1% | 27% | 0.2% | 0/0 | 1% |
| `ga_surrogate_cifar` | 5 | 3% | 49% | 6.4% | 6/7 | 19% |
| `ga_surrogate_nopool_cifar` | 5 | 1% | 33% | 6.2% | 7/7 | 23% |
| `ga_perclient_cifar_r40` | 3 | 1% | 45% | 1.2% | 0/0 | 4% |
| `ga_perclient_alpha01` | 3 | 1% | 32% | 0.7% | 0/1 | 4% |
| `ga_surrogate_alpha01` | 3 | 1% | 56% | 12.3% | 6/9 | 33% |

## Busca por broadcast (uma configuração por round para todos)

| Cenário | Runs | Rounds com configuração nociva | Quedas | Quedas em round nocivo | Rounds até voltar a 5 pp do valor anterior |
|---|---:|---:|---:|---:|---|
| `ga_broadcast_cifar` | 5 | 13/100 | 4 | 4/4 | 2, 1, 2, 1 |
| `ga_broadcast_noelite_cifar` | 5 | 11/100 | 2 | 2/2 | 2, 1 |
| `ga_broadcast_cifar_r40` | 5 | 15/200 | 2 | 2/2 | 1, 1 |
| `tpe_broadcast_cifar` | 5 | 16/100 | 4 | 3/4 | 1, 1, 1, 1 |
| `rs_broadcast_cifar` | 5 | 28/100 | 10 | 10/10 | 2, 2, 1, 1, —, 1, 1, 1, 1, 1 |
