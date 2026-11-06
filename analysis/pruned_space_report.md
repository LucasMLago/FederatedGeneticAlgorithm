# Pruned search space against the full space

_Source: `matrix_summary_pruned.csv` (pruned) and `matrix_summary_final.csv` (full space), seeds 0-4, same protocol. Severe drop: more than 10 pp in one round. N = 5 per cell, so the comparison is descriptive._

**Reading: in between: part of the drops comes from the harmful region, part from configurations near its border.** Pooled over the three searchers, the full space had 18 drops in 12/15 runs and the pruned space 7 in 6/15.

| Searcher | Space | Runs | Drops per run (seeds 0-4) | Total | Runs with a drop | Peak % | Final % | Redrawn proposals |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Broadcast GA | full | 5 | 1, 2, 0, 1, 0 | 4 | 3 | 79.1 +- 1.9 | 78.2 +- 2.2 | - |
| Broadcast GA | pruned | 5 | 0, 0, 0, 1, 1 | 2 | 2 | 79.9 +- 2.3 | 78.8 +- 2.8 | 11 |
| RS | full | 5 | 2, 3, 2, 2, 1 | 10 | 5 | 76.1 +- 3.9 | 65.4 +- 20.7 | - |
| RS | pruned | 5 | 1, 0, 0, 0, 0 | 1 | 1 | 80.1 +- 2.3 | 79.0 +- 4.1 | 35 |
| TPE | full | 5 | 1, 1, 1, 0, 1 | 4 | 4 | 78.6 +- 3.5 | 78.1 +- 3.2 | - |
| TPE | pruned | 5 | 1, 0, 0, 2, 1 | 4 | 3 | 80.4 +- 1.0 | 79.9 +- 1.3 | 0 |
| Per-client GA | full | 5 | 0, 0, 0, 0, 0 | 0 | 0 | 83.2 +- 0.7 | 82.8 +- 1.2 | - |

Gap to per-client GA on the same seeds (pruned searcher minus per-client, pp):

| Searcher | Peak | Final | Closed (within 1 pp on both) |
|---|---:|---:|---|
| Broadcast GA | -3.3 | -4.0 | no |
| RS | -3.1 | -3.8 | no |
| TPE | -2.9 | -2.8 | no |
