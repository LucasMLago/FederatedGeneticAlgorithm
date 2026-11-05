# Ceiling probe: fixed SGD variants of the expert

_Source: `federatedgeneticalgorithm/artifacts/matrix_summary_probes.csv` for the probes, `federatedgeneticalgorithm/artifacts/matrix_summary_final.csv` for the expert. Seed 0, one run per probe._

**The best probe is `probe_sgd_bs128` (batch 128), peak 84.93%, +0.27 pp against the expert's 84.66% on seed 0, which is within the expert's seed-to-seed sd (0.56 pp over 10 seeds).** Probe peaks range from 82.49 to 84.93%; 5 probes are inside the grid and 3 outside it.

| Probe | Change from the expert | In the grid | Peak % | Final % | Wall (min) | run_id |
|---|---|---|---:|---:|---:|---|
| expert, seed 0 | none | yes | 84.66 | 83.37 | 49.7 | `20260519_020412` |
| `probe_sgd_bs128` | batch 128 | yes | 84.93 | 84.93 | 48.5 | `20261026_203025` |
| `probe_sgd_lr0p005` | lr 0.005 | yes | 84.78 | 83.32 | 49.9 | `20261026_211853` |
| `probe_sgd_wd1e-3` | wd 0.001 | yes | 84.37 | 83.64 | 49.8 | `20261027_003247` |
| `probe_sgd_wd5e-4` | wd 0.0005 | no | 84.29 | 84.23 | 53.5 | `20261027_020759` |
| `probe_sgd_wd1e-4` | wd 0.0001 | yes | 83.88 | 81.89 | 45.4 | `20261027_012235` |
| `probe_sgd_lr0p02_wd5e-4` | lr 0.02, wd 0.0005 | no | 83.07 | 83.07 | 48.3 | `20261026_220850` |
| `probe_sgd_lr0p02` | lr 0.02 | no | 82.63 | 81.78 | 43.3 | `20261026_225707` |
| `probe_sgd_mom0p95` | momentum 0.95 | yes | 82.49 | 81.46 | 52.4 | `20261026_234026` |
