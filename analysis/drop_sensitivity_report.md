# Sensibilidade do limiar de queda severa

_Fonte: `federatedgeneticalgorithm/artifacts/matrix_summary_final.csv` e `server_aggregated_rounds.csv` de cada run. Queda = perda de eval-acc entre dois rounds seguidos maior que o limiar. Cada célula traz o total de quedas e, entre parênteses, quantas runs tiveram pelo menos uma. A 10 pp os números são os do artigo (conferido contra `results_tables.py`)._

## Cenário principal (CIFAR-10, ResNet-18, alpha = 0,5, 20 rounds)

| Desenho | N | > 5 pp | > 10 pp | > 20 pp | Maior queda por run, média (pp) | Peak - final, média (pp) |
|---|---:|---:|---:|---:|---:|---:|
| fixed HP (expert) (`fixed_expert_cifar`) | 10 | 0 (0/10) | 0 (0/10) | 0 (0/10) | 1.4 | 0.6 |
| fixed HP (naive) (`fixed_naive_cifar`) | 10 | 4 (3/10) | 0 (0/10) | 0 (0/10) | 4.1 | 0.1 |
| GA zero-coupling (`ga_perclient_cifar`) | 10 | 4 (2/10) | 1 (1/10) | 0 (0/10) | 3.0 | 0.9 |
| FedPop (`fedpop_cifar`) | 10 | 1 (1/10) | 1 (1/10) | 0 (0/10) | 3.3 | 0.3 |
| FedEx (`fedex_cifar`) | 10 | 3 (3/10) | 2 (2/10) | 0 (0/10) | 4.4 | 0.6 |
| GA medium-coupling (`ga_surrogate_cifar`) | 10 | 13 (7/10) | 9 (4/10) | 3 (2/10) | 14.7 | 5.5 |
| GA high-coupling (`ga_broadcast_cifar`) | 10 | 19 (9/10) | 11 (7/10) | 7 (4/10) | 27.7 | 6.2 |
| RS high-coupling (`rs_broadcast_cifar`) | 10 | 35 (10/10) | 17 (9/10) | 11 (8/10) | 31.1 | 9.1 |
| TPE high-coupling (`tpe_broadcast_cifar`) | 10 | 26 (10/10) | 11 (8/10) | 2 (2/10) | 15.6 | 1.4 |

## Ordem entre os três regimes do GA

- > 5 pp: runs com queda per-client 2, surrogate 7, broadcast 9 (quedas: 4, 13, 19). per-client <= surrogate <= broadcast: sim.
- > 10 pp: runs com queda per-client 1, surrogate 4, broadcast 7 (quedas: 1, 9, 11). per-client <= surrogate <= broadcast: sim.
- > 20 pp: runs com queda per-client 0, surrogate 2, broadcast 4 (quedas: 0, 3, 7). per-client <= surrogate <= broadcast: sim.

## Demais cenários do artigo

| Família | Desenho | N | > 5 pp | > 10 pp | > 20 pp |
|---|---|---:|---:|---:|---:|
| femnist | fixed HP (expert) (`fixed_expert_femnist`) | 3 | 0 (0/3) | 0 (0/3) | 0 (0/3) |
| femnist | fixed HP (naive) (`fixed_naive_femnist`) | 3 | 0 (0/3) | 0 (0/3) | 0 (0/3) |
| femnist | GA zero-coupling (`ga_perclient_femnist`) | 3 | 0 (0/3) | 0 (0/3) | 0 (0/3) |
| femnist | GA medium-coupling (`ga_surrogate_femnist`) | 3 | 2 (2/3) | 1 (1/3) | 0 (0/3) |
| femnist | GA high-coupling (`ga_broadcast_femnist`) | 3 | 2 (1/3) | 1 (1/3) | 1 (1/3) |
| femnist | RS high-coupling (`rs_broadcast_femnist`) | 3 | 4 (3/3) | 3 (3/3) | 2 (2/3) |
| femnist | TPE high-coupling (`tpe_broadcast_femnist`) | 3 | 2 (2/3) | 2 (2/3) | 0 (0/3) |
| cifar-small | fixed HP (expert) (`fixed_expert_smallcnn`) | 3 | 0 (0/3) | 0 (0/3) | 0 (0/3) |
| cifar-small | fixed HP (naive) (`fixed_naive_smallcnn`) | 3 | 0 (0/3) | 0 (0/3) | 0 (0/3) |
| cifar-small | GA zero-coupling (`ga_perclient_smallcnn`) | 3 | 1 (1/3) | 0 (0/3) | 0 (0/3) |
| cifar-small | GA medium-coupling (`ga_surrogate_smallcnn`) | 3 | 1 (1/3) | 0 (0/3) | 0 (0/3) |
| cifar-small | GA high-coupling (`ga_broadcast_smallcnn`) | 3 | 2 (2/3) | 0 (0/3) | 0 (0/3) |
| cifar-alpha01 | fixed HP (expert) (`fixed_expert_alpha01`) | 3 | 7 (3/3) | 1 (1/3) | 0 (0/3) |
| cifar-alpha01 | fixed HP (naive) (`fixed_naive_alpha01`) | 3 | 9 (3/3) | 4 (2/3) | 0 (0/3) |
| cifar-alpha01 | GA zero-coupling (`ga_perclient_alpha01`) | 3 | 8 (3/3) | 1 (1/3) | 0 (0/3) |
| cifar-alpha01 | GA medium-coupling (`ga_surrogate_alpha01`) | 3 | 13 (3/3) | 9 (3/3) | 5 (2/3) |
| cifar-alpha01 | GA high-coupling (`ga_broadcast_alpha01`) | 3 | 10 (3/3) | 6 (3/3) | 1 (1/3) |
| failure-modes | broadcast GA, random-init pop (`ga_broadcast_randominit_cifar`) | 3 | 7 (3/3) | 6 (3/3) | 1 (1/3) |
| failure-modes | broadcast GA, delta fitness (`ga_broadcast_deltafitness_cifar`) | 3 | 6 (3/3) | 5 (3/3) | 2 (1/3) |
| horizon | per-client GA, 40 rounds (`ga_perclient_cifar_r40`) | 3 | 2 (1/3) | 0 (0/3) | 0 (0/3) |
| horizon | broadcast GA, 40 rounds (`ga_broadcast_cifar_r40`) | 5 | 10 (5/5) | 2 (2/5) | 2 (2/5) |
| horizon | fixed HP (expert), 40 rounds (`fixed_expert_cifar_r40`) | 5 | 0 (0/5) | 0 (0/5) | 0 (0/5) |
| ablations | surrogate GA, no shared pool (`ga_surrogate_nopool_cifar`) | 5 | 10 (4/5) | 7 (4/5) | 1 (1/5) |
| ablations | broadcast GA, no elitism (`ga_broadcast_noelite_cifar`) | 5 | 5 (4/5) | 2 (2/5) | 2 (2/5) |
| ablations | surrogate GA, long evaluation (`ga_surrogate_longeval_cifar`) | 5 | 1 (1/5) | 0 (0/5) | 0 (0/5) |
