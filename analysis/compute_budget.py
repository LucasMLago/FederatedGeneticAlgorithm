#!/usr/bin/env python3
"""Client compute per run, split into federated training and hyperparameter search.

Reads client_round_metrics.csv of every run. Search cost is converted to local-epoch equivalents
with the run's own training speed: one epoch = total local training time / (visits x LOCAL_EPOCHS),
so a run's search time divided by that is how many federated-training epochs it could have bought.

Usage:
    python analysis/compute_budget.py
    python analysis/compute_budget.py --markdown analysis/compute_budget_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from results_tables import FAMILIES, REPO_ROOT, RUNS_DIR, SUMMARY, load_summary

ORDER = [
    "fixed_expert_cifar", "ga_broadcast_cifar", "rs_broadcast_cifar", "tpe_broadcast_cifar", "fedex_cifar",
    "ga_surrogate_cifar", "ga_perclient_cifar", "fedpop_cifar",
    "ga_surrogate_nopool_cifar", "ga_surrogate_longeval_cifar",
    "fixed_expert_cifar_r40", "ga_broadcast_cifar_r40", "ga_perclient_cifar_r40",
]


def run_cost(run_id: str) -> dict:
    d = RUNS_DIR / run_id
    local_epochs = json.loads((d / "resolved_config.json").read_text(encoding="utf-8"))["LOCAL_EPOCHS"]
    visits = train_s = search_s = 0
    with (d / "client_round_metrics.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            visits += 1
            train_s += float(r["local_train_time_s"] or 0)
            search_s += float(r["ga_time_s"] or 0)
    train_epochs = visits * local_epochs
    epoch_s = train_s / train_epochs
    return {"visits": visits, "train_epochs": train_epochs, "train_min": train_s / 60,
            "search_min": search_s / 60, "search_epochs": search_s / epoch_s}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    by_scn: dict[str, list[dict]] = defaultdict(list)
    for r in load_summary():
        if r["scenario_name"] in ORDER:
            r.update(run_cost(r["run_id"]))
            by_scn[r["scenario_name"]].append(r)

    def m(rs, k):
        return statistics.fmean(r[k] for r in rs)

    lines = ["# Orçamento de computação por run\n",
             f"_Fonte: `{SUMMARY.relative_to(REPO_ROOT)}` e `client_round_metrics.csv` de cada run. Médias por "
             "run. Épocas de treino = visitas de clientes x LOCAL_EPOCHS (8). Busca em épocas equivalentes = tempo "
             "de busca nos clientes dividido pelo tempo de uma época de treino federado da mesma run. A busca do "
             "lado do servidor (broadcast, RS, TPE, FedEx) não roda nos clientes e fica fora da conta._\n"]
    lines.append("| Desenho | N | Visitas | Épocas de treino | Busca (min) | Busca (épocas equiv.) | Total (épocas equiv.) | Busca / total | Wall (min) | Peak % | Final % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    totals = {}
    for scn in ORDER:
        rs = by_scn.get(scn)
        if not rs:
            continue
        tr, se = m(rs, "train_epochs"), m(rs, "search_epochs")
        totals[scn] = (tr + se, m(rs, "peak") * 100, m(rs, "final") * 100)
        lines.append(
            f"| {FAMILIES[scn][1]} (`{scn}`) | {len(rs)} | {m(rs, 'visits'):.0f} | {tr:.0f} | {m(rs, 'search_min'):.1f} "
            f"| {se:.0f} | {tr + se:.0f} | {se / (tr + se) * 100:.0f}% | {m(rs, 'wall_min'):.1f} "
            f"| {m(rs, 'peak') * 100:.1f} | {m(rs, 'final') * 100:.1f} |")

    if {"ga_perclient_cifar", "ga_broadcast_cifar_r40"} <= totals.keys():
        pc, bc = totals["ga_perclient_cifar"], totals["ga_broadcast_cifar_r40"]
        lines.append("")
        lines.append(f"- Broadcast GA com 40 rounds usou {bc[0]:.0f} épocas equivalentes, {bc[0] / pc[0] * 100:.0f}% das "
                     f"{pc[0]:.0f} do per-client com 20 rounds, e chegou a peak {bc[1]:.1f}% contra {pc[1]:.1f}% "
                     f"(final {bc[2]:.1f}% contra {pc[2]:.1f}%). N = 5 e 10, seeds diferentes em parte.")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.markdown:
        args.markdown.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
