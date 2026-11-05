#!/usr/bin/env python3
"""Holm and Benjamini-Hochberg adjustment of the ten-seed tests reported in Section V-A.

The family is the nine peak-accuracy comparisons of the main scenario that the paper reports with a
p-value. Each p is recomputed with results_tables.pairwise() (exact Wilcoxon signed-rank on the ten
paired seeds) and shown next to the value printed in the paper.

Usage:
    python analysis/multiple_testing.py
    python analysis/multiple_testing.py --markdown analysis/multiple_testing_report.md
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from results_tables import REPO_ROOT, SUMMARY, load_summary, pairwise

ALPHA = 0.05

# (label, a, b, p as printed in the paper)
FAMILY = [
    ("per-client vs FedPop", "ga_perclient_cifar", "fedpop_cifar", "0.28"),
    ("per-client vs broadcast GA", "ga_perclient_cifar", "ga_broadcast_cifar", "0.002"),
    ("per-client vs surrogate", "ga_perclient_cifar", "ga_surrogate_cifar", "0.020"),
    ("per-client vs FedEx", "ga_perclient_cifar", "fedex_cifar", "0.020"),
    ("expert vs per-client", "fixed_expert_cifar", "ga_perclient_cifar", "0.002"),
    ("expert vs FedPop", "fixed_expert_cifar", "fedpop_cifar", "0.049"),
    ("FedEx vs broadcast GA", "fedex_cifar", "ga_broadcast_cifar", "0.014"),
    ("FedPop vs FedEx", "fedpop_cifar", "fedex_cifar", "0.006"),
    ("broadcast GA vs naive", "ga_broadcast_cifar", "fixed_naive_cifar", "0.70"),
]


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def bh(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i], reverse=True)
    adj = [0.0] * m
    running = 1.0
    for k, i in enumerate(order):
        rank = m - k
        running = min(running, ps[i] * m / rank)
        adj[i] = running
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    by_scn: dict[str, list[dict]] = defaultdict(list)
    for r in load_summary():
        by_scn[r["scenario_name"]].append(r)

    res = []
    for label, a, b, paper in FAMILY:
        pk = pairwise(by_scn, a, b, "peak")
        if pk is None or pk["wil_p"] is None or pk["n_paired"] != 10:
            print(f"{label}: expected ten paired seeds, got {pk}", file=sys.stderr)
            return 1
        res.append((label, pk["delta_pp"], paper, pk["wil_p"]))

    ps = [p for *_, p in res]
    h, f = holm(ps), bh(ps)
    lines = ["# Ajuste para comparações múltiplas (cenário principal, 10 seeds)\n",
             f"_Fonte: `{SUMMARY.relative_to(REPO_ROOT)}`. Família: as 9 comparações de peak da §V-A com p-valor "
             "no artigo. p recalculado com Wilcoxon exato pareado (`results_tables.pairwise`). Holm controla a "
             "chance de pelo menos um falso positivo na família; Benjamini-Hochberg controla a proporção esperada "
             f"de falsos positivos entre as diferenças declaradas. Nível {ALPHA}._\n"]
    lines.append("| Comparação | Δpeak (pp, a - b) | p no artigo | p recalculado | p Holm | p BH | Significativo sem ajuste / Holm / BH |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for (label, d, paper, p), ph, pf in zip(res, h, f):
        flags = " / ".join("sim" if x <= ALPHA else "não" for x in (p, ph, pf))
        lines.append(f"| {label} | {d:+.2f} | {paper} | {p:.4f} | {ph:.4f} | {pf:.4f} | {flags} |")

    lost_h = [r[0] for r, p, ph in zip(res, ps, h) if p <= ALPHA < ph]
    lost_f = [r[0] for r, p, pf in zip(res, ps, f) if p <= ALPHA < pf]
    lines.append("")
    lines.append(f"- Deixam de ser significativas com Holm: {', '.join(lost_h) if lost_h else 'nenhuma'}.")
    lines.append(f"- Deixam de ser significativas com BH: {', '.join(lost_f) if lost_f else 'nenhuma'}.")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.markdown:
        args.markdown.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
