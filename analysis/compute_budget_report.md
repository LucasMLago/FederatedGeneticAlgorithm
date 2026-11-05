# Orçamento de computação por run

_Fonte: `federatedgeneticalgorithm/artifacts/matrix_summary_final.csv` e `client_round_metrics.csv` de cada run. Médias por run. Épocas de treino = visitas de clientes x LOCAL_EPOCHS (8). Busca em épocas equivalentes = tempo de busca nos clientes dividido pelo tempo de uma época de treino federado da mesma run. A busca do lado do servidor (broadcast, RS, TPE, FedEx) não roda nos clientes e fica fora da conta._

| Desenho | N | Visitas | Épocas de treino | Busca (min) | Busca (épocas equiv.) | Total (épocas equiv.) | Busca / total | Wall (min) | Peak % | Final % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed HP (expert) (`fixed_expert_cifar`) | 10 | 100 | 800 | 0.0 | 0 | 800 | 0% | 50.2 | 84.4 | 83.7 |
| GA high-coupling (`ga_broadcast_cifar`) | 10 | 100 | 800 | 0.0 | 0 | 800 | 0% | 51.3 | 78.6 | 72.5 |
| RS high-coupling (`rs_broadcast_cifar`) | 10 | 100 | 800 | 0.0 | 0 | 800 | 0% | 51.9 | 74.4 | 65.3 |
| TPE high-coupling (`tpe_broadcast_cifar`) | 10 | 100 | 800 | 0.0 | 0 | 800 | 0% | 50.9 | 77.2 | 75.8 |
| FedEx (`fedex_cifar`) | 10 | 100 | 800 | 0.0 | 0 | 800 | 0% | 50.1 | 81.1 | 80.5 |
| GA medium-coupling (`ga_surrogate_cifar`) | 10 | 100 | 800 | 27.9 | 268 | 1068 | 25% | 64.1 | 79.8 | 74.2 |
| GA zero-coupling (`ga_perclient_cifar`) | 10 | 100 | 800 | 163.1 | 1536 | 2336 | 66% | 141.6 | 82.7 | 81.7 |
| FedPop (`fedpop_cifar`) | 10 | 500 | 4000 | 0.0 | 0 | 4000 | 0% | 249.2 | 83.2 | 83.0 |
| surrogate GA, no shared pool (`ga_surrogate_nopool_cifar`) | 5 | 100 | 800 | 24.6 | 235 | 1035 | 23% | 62.4 | 81.1 | 79.4 |
| surrogate GA, long evaluation (`ga_surrogate_longeval_cifar`) | 5 | 100 | 800 | 183.4 | 1733 | 2533 | 68% | 149.0 | 83.2 | 82.0 |
| fixed HP (expert), 40 rounds (`fixed_expert_cifar_r40`) | 5 | 200 | 1600 | 0.0 | 0 | 1600 | 0% | 97.9 | 86.3 | 86.0 |
| broadcast GA, 40 rounds (`ga_broadcast_cifar_r40`) | 5 | 200 | 1600 | 0.0 | 0 | 1600 | 0% | 101.8 | 83.7 | 82.5 |
| per-client GA, 40 rounds (`ga_perclient_cifar_r40`) | 3 | 200 | 1600 | 269.5 | 2505 | 4105 | 61% | 252.3 | 85.5 | 85.0 |

- Broadcast GA com 40 rounds usou 1600 épocas equivalentes, 69% das 2336 do per-client com 20 rounds, e chegou a peak 83.7% contra 82.7% (final 82.5% contra 81.7%). N = 5 e 10, seeds diferentes em parte.
