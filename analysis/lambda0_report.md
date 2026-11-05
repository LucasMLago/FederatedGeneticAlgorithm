# Per-client GA without the drift penalty

_Source: `federatedgeneticalgorithm/artifacts/matrix_summary_final.csv` and `federatedgeneticalgorithm/artifacts/matrix_summary_probes.csv`, with the per-run telemetry. Severe drop: more than 10 pp in one round. Harmful: lion with lr >= 5e-3, or adam, adamw, radam with lr = 1e-2._

| Arm | Runs | Peak % | Final % | Severe drops (runs with one) | Harmful updates |
|---|---:|---:|---:|---:|---:|
| per-client GA, seeds 0-2 | 3 | 83.52 +- 0.70 | 82.78 +- 1.67 | 0 (0/3) | 0.3% (1/300) |
| per-client GA, all ten seeds | 10 | 82.68 +- 1.16 | 81.75 +- 2.26 | 1 (1/10) | 1.2% (12/1000) |
| per-client GA, lambda = 0, seeds 0-2 | 3 | 83.17 +- 0.52 | 82.74 +- 0.73 | 0 (0/3) | 5.7% (17/300) |

Lambda-0 runs, one by one:

| Seed | Peak % | Final % | Severe drops | Harmful updates |
|---:|---:|---:|---:|---:|
| 0 | 83.37 | 82.40 | 0 | 0/100 |
| 1 | 83.57 | 83.57 | 0 | 17/100 |
| 2 | 82.58 | 82.24 | 0 | 0/100 |
