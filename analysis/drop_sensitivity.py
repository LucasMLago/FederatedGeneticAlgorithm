#!/usr/bin/env python3
"""How the severe-drop counts change with the threshold (5, 10 and 20 pp in one round).

Uses the same per-round eval-acc curves as results_tables.py; at 10 pp the counts must match its
robustness() exactly, and the script stops if they don't.

Usage:
    python analysis/drop_sensitivity.py
    python analysis/drop_sensitivity.py --markdown analysis/drop_sensitivity_report.md
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from results_tables import FAMILIES, REPO_ROOT, SUMMARY, eval_curve, load_summary, robustness

THRESHOLDS = (0.05, 0.10, 0.20)
PAPER_THRESHOLD = 0.10

MAIN_ORDER = [
    "fixed_expert_cifar", "fixed_naive_cifar", "ga_perclient_cifar", "fedpop_cifar", "fedex_cifar",
    "ga_surrogate_cifar", "ga_broadcast_cifar", "rs_broadcast_cifar", "tpe_broadcast_cifar",
]
GA_REGIMES = ["ga_perclient_cifar", "ga_surrogate_cifar", "ga_broadcast_cifar"]


def drops(curve: list[float | None], threshold: float) -> int:
    """Single-round decreases larger than threshold, skipping rounds without eval-acc (as robustness())."""
    vals = [v for v in curve if v is not None]
    return sum(1 for prev, cur in zip(vals, vals[1:]) if prev - cur > threshold)


def max_drop(curve: list[float | None]) -> float:
    vals = [v for v in curve if v is not None]
    return max([prev - cur for prev, cur in zip(vals, vals[1:])] + [0.0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    rows = load_summary()
    by_scn: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["scenario_name"] in FAMILIES:
            curve = eval_curve(r["run_id"])
            r["curve"] = curve
            r["drops"] = {t: drops(curve, t) for t in THRESHOLDS}
            r["max_drop"] = max_drop(curve)
            if r["drops"][PAPER_THRESHOLD] != robustness(curve)["severe_drops"]:
                print(f"mismatch with results_tables at 10 pp: {r['run_id']}", file=sys.stderr)
                return 1
            by_scn[r["scenario_name"]].append(r)

    def cell(rs: list[dict], t: float) -> str:
        total = sum(r["drops"][t] for r in rs)
        runs = sum(1 for r in rs if r["drops"][t] > 0)
        return f"{total} ({runs}/{len(rs)})"

    lines = ["# Sensibilidade do limiar de queda severa\n",
             f"_Fonte: `{SUMMARY.relative_to(REPO_ROOT)}` e `server_aggregated_rounds.csv` de cada run. "
             "Queda = perda de eval-acc entre dois rounds seguidos maior que o limiar. Cada célula traz o total "
             "de quedas e, entre parênteses, quantas runs tiveram pelo menos uma. A 10 pp os números são os do "
             "artigo (conferido contra `results_tables.py`)._\n"]

    lines.append("## Cenário principal (CIFAR-10, ResNet-18, alpha = 0,5, 20 rounds)\n")
    lines.append("| Desenho | N | > 5 pp | > 10 pp | > 20 pp | Maior queda por run, média (pp) | Peak - final, média (pp) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for scn in MAIN_ORDER:
        rs = by_scn.get(scn, [])
        if not rs:
            continue
        md = statistics.fmean(r["max_drop"] for r in rs) * 100
        gap = statistics.fmean(r["peak"] - r["final"] for r in rs) * 100
        lines.append(f"| {FAMILIES[scn][1]} (`{scn}`) | {len(rs)} | "
                     + " | ".join(cell(rs, t) for t in THRESHOLDS) + f" | {md:.1f} | {gap:.1f} |")

    lines.append("\n## Ordem entre os três regimes do GA\n")
    for t in THRESHOLDS:
        runs = {s: sum(1 for r in by_scn[s] if r["drops"][t] > 0) for s in GA_REGIMES}
        total = {s: sum(r["drops"][t] for r in by_scn[s]) for s in GA_REGIMES}
        holds = runs["ga_perclient_cifar"] <= runs["ga_surrogate_cifar"] <= runs["ga_broadcast_cifar"]
        lines.append(f"- > {t * 100:.0f} pp: runs com queda per-client {runs['ga_perclient_cifar']}, surrogate "
                     f"{runs['ga_surrogate_cifar']}, broadcast {runs['ga_broadcast_cifar']} (quedas: "
                     f"{total['ga_perclient_cifar']}, {total['ga_surrogate_cifar']}, {total['ga_broadcast_cifar']}). "
                     f"per-client <= surrogate <= broadcast: {'sim' if holds else 'não'}.")

    lines.append("\n## Demais cenários do artigo\n")
    lines.append("| Família | Desenho | N | > 5 pp | > 10 pp | > 20 pp |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for scn, (fam, label) in FAMILIES.items():
        if scn in MAIN_ORDER or scn not in by_scn:
            continue
        rs = by_scn[scn]
        lines.append(f"| {fam} | {label} (`{scn}`) | {len(rs)} | " + " | ".join(cell(rs, t) for t in THRESHOLDS) + " |")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.markdown:
        args.markdown.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
