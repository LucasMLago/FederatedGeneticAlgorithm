# Ajuste para comparações múltiplas (cenário principal, 10 seeds)

_Fonte: `federatedgeneticalgorithm/artifacts/matrix_summary_final.csv`. Família: as 9 comparações de peak da §V-A com p-valor no artigo. p recalculado com Wilcoxon exato pareado (`results_tables.pairwise`). Holm controla a chance de pelo menos um falso positivo na família; Benjamini-Hochberg controla a proporção esperada de falsos positivos entre as diferenças declaradas. Nível 0.05._

| Comparação | Δpeak (pp, a - b) | p no artigo | p recalculado | p Holm | p BH | Significativo sem ajuste / Holm / BH |
|---|---:|---:|---:|---:|---:|---|
| per-client vs FedPop | -0.54 | 0.28 | 0.2754 | 0.5508 | 0.3098 | não / não / não |
| per-client vs broadcast GA | +4.03 | 0.002 | 0.0020 | 0.0176 | 0.0088 | sim / sim / sim |
| per-client vs surrogate | +2.90 | 0.020 | 0.0195 | 0.0977 | 0.0293 | sim / não / sim |
| per-client vs FedEx | +1.59 | 0.020 | 0.0195 | 0.0977 | 0.0293 | sim / não / sim |
| expert vs per-client | +1.68 | 0.002 | 0.0020 | 0.0176 | 0.0088 | sim / sim / sim |
| expert vs FedPop | +1.14 | 0.049 | 0.0488 | 0.1465 | 0.0628 | sim / não / não |
| FedEx vs broadcast GA | +2.45 | 0.014 | 0.0137 | 0.0820 | 0.0293 | sim / não / sim |
| FedPop vs FedEx | +2.13 | 0.006 | 0.0059 | 0.0410 | 0.0176 | sim / sim / sim |
| broadcast GA vs naive | +0.54 | 0.70 | 0.6953 | 0.6953 | 0.6953 | não / não / não |

- Deixam de ser significativas com Holm: per-client vs surrogate, per-client vs FedEx, expert vs FedPop, FedEx vs broadcast GA.
- Deixam de ser significativas com BH: expert vs FedPop.
