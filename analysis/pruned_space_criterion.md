# Pruned search space: how the result will be read

_Written on 2026-11-05, before any pruned run was launched._

**Question.** Part of the instability of the broadcast searchers may come from a harmful region that the search space contains. If the searchers never train a harmful configuration, do the severe drops go away, and does the gap to per-client search close?

**Runs.** Broadcast GA, RS and TPE with `FED_EXCLUDE_HARMFUL=true` (configs `*_broadcast_pruned_cifar.yaml`), CIFAR-10, ResNet-18, alpha = 0.5, 20 rounds, seeds 0-4, same protocol as the full-space runs. 15 runs.

**Reference (full space, same seeds 0-4, severe drop = more than 10 pp in one round).**

| Searcher | Drops per run (seeds 0-4) | Total | Runs with a drop | Peak % | Final % |
|---|---|---:|---:|---:|---:|
| Broadcast GA | 1, 2, 0, 1, 0 | 4 | 3/5 | 79.1 +- 1.9 | 78.2 +- 2.2 |
| RS | 2, 3, 2, 2, 1 | 10 | 5/5 | 76.1 +- 3.9 | 65.4 +- 20.7 |
| TPE | 1, 1, 1, 0, 1 | 4 | 4/5 | 78.6 +- 3.5 | 78.1 +- 3.2 |
| All three | | 18 | 12/15 | | |
| Per-client GA (for the gap) | 0, 0, 0, 0, 0 | 0 | 0/5 | 83.2 +- 0.7 | 82.8 +- 1.2 |

**Reading, fixed now, on the 15 pruned runs taken together.**

- Most of the instability comes from the harmful region: at most 6 severe drops (a third of 18) and at most 5 of 15 runs with a drop.
- The instability does not come mainly from the harmful region: at least 12 severe drops (two thirds of 18) and at least 10 of 15 runs with a drop.
- Anything in between: part of the drops comes from the harmful region and part from configurations next to its border. This is the outcome the earlier seeds 5-9 suggest (6 of 21 drops followed learning rates just below the harmful set).
- The two counts must point the same way for the first two readings; if they disagree the reading is "in between".

**Gap to per-client search.** Reported as the difference in mean peak and mean final accuracy between each pruned searcher and per-client GA on seeds 0-4. The gap counts as closed only if a pruned searcher is within 1 pp of per-client GA on both.

**What is reported either way.** Per searcher and pooled: drops per run, runs with a drop, peak and final (mean +- sd), and how many proposals the exclusion redrew. With N = 5 the smallest attainable Wilcoxon p is 0.0625, so no significance claim is made; the comparison is descriptive.
