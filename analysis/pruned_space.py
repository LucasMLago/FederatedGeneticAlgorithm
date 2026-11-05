#!/usr/bin/env python3
"""Broadcast searchers on the pruned search space against the same searchers on the full space.

The pruned runs (`*_broadcast_pruned_cifar`, seeds 0-4) never train a harmful configuration. The
script compares them with the full-space runs of the same seeds: severe drops (more than 10 pp in one
round), peak and final accuracy, and how many proposals the exclusion redrew. The verdict follows the
reading fixed before the runs: pooled over the three searchers, at most 6 drops and at most 5 of 15
runs with a drop mean the harmful region explains most of the instability; at least 12 drops and at
least 10 runs mean it does not; anything else, or counts that disagree, is "in between". Per-client
GA on the same seeds is the reference for the accuracy gap, closed only within 1 pp on peak and final.

Usage:
    python analysis/pruned_space.py
    python analysis/pruned_space.py --markdown analysis/pruned_space_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

import client_agreement as ca

REPO_ROOT = ca.REPO_ROOT
ARTIFACTS = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts"
FINAL = ca.SUMMARY
PRUNED = Path(os.environ.get("FGA_PRUNED", ARTIFACTS / "matrix_summary_pruned.csv"))
SEEDS = range(5)
SEARCHERS = [
    ("Broadcast GA", "ga_broadcast_cifar", "ga_broadcast_pruned_cifar"),
    ("RS", "rs_broadcast_cifar", "rs_broadcast_pruned_cifar"),
    ("TPE", "tpe_broadcast_cifar", "tpe_broadcast_pruned_cifar"),
]
REFERENCE = "ga_perclient_cifar"
FEWER, MORE = (6, 5), (12, 10)  # (drops, runs with a drop) bounds of the pre-registered reading
GAP_CLOSED_PP = 1.0


def rows(path: Path, scenario: str) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return sorted((r for r in csv.DictReader(fh) if r["scenario_name"] == scenario and r["status"] == "ok"
                       and int(r["seed"]) in SEEDS), key=lambda r: int(r["seed"]))


def summarize(rs: list[dict]) -> dict:
    drops = [len(ca.drops(ca.eval_acc(r["run_id"]))) for r in rs]
    peaks = [100 * float(r["peak_eval_acc"]) for r in rs]
    finals = [100 * float(r["final_eval_acc"]) for r in rs]
    rejected = 0
    for r in rs:
        f = ca.RUNS_DIR / r["run_id"] / "searcher_rejections.json"
        if f.exists():
            rejected += json.loads(f.read_text(encoding="utf-8")).get("rejected", 0)
    sd = lambda v: statistics.stdev(v) if len(v) > 1 else float("nan")
    return {"n": len(rs), "drops": drops, "total": sum(drops), "with": sum(d > 0 for d in drops),
            "peak": statistics.fmean(peaks) if peaks else float("nan"), "peak_sd": sd(peaks),
            "final": statistics.fmean(finals) if finals else float("nan"), "final_sd": sd(finals),
            "rejected": rejected}


def verdict(total: int, runs_with: int, n: int) -> str:
    if n < 15:
        return f"incomplete: {n} of 15 pruned runs"
    if total <= FEWER[0] and runs_with <= FEWER[1]:
        return "most of the instability comes from the harmful region"
    if total >= MORE[0] and runs_with >= MORE[1]:
        return "the instability does not come mainly from the harmful region"
    return "in between: part of the drops comes from the harmful region, part from configurations near its border"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    full = {name: summarize(rows(FINAL, scn)) for name, scn, _ in SEARCHERS}
    pruned = {name: summarize(rows(PRUNED, scn)) for name, _, scn in SEARCHERS}
    ref = summarize(rows(FINAL, REFERENCE))
    pool = lambda arms, key: sum(a[key] for a in arms.values())

    lines = ["# Pruned search space against the full space", ""]
    lines.append(f"_Source: `{PRUNED.name}` (pruned) and `{FINAL.name}` (full space), seeds 0-4, same protocol. "
                 f"Severe drop: more than {ca.SEVERE_DROP:.0f} pp in one round. N = 5 per cell, so the "
                 f"comparison is descriptive._\n")
    lines.append(f"**Reading: {verdict(pool(pruned, 'total'), pool(pruned, 'with'), pool(pruned, 'n'))}.** "
                 f"Pooled over the three searchers, the full space had {pool(full, 'total')} drops in "
                 f"{pool(full, 'with')}/{pool(full, 'n')} runs and the pruned space "
                 f"{pool(pruned, 'total')} in {pool(pruned, 'with')}/{pool(pruned, 'n')}.\n")
    lines.append("| Searcher | Space | Runs | Drops per run (seeds 0-4) | Total | Runs with a drop | Peak % | Final % | Redrawn proposals |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for name, _, _ in SEARCHERS:
        for label, arm in (("full", full[name]), ("pruned", pruned[name])):
            lines.append(f"| {name} | {label} | {arm['n']} | {', '.join(map(str, arm['drops'])) or '-'} | {arm['total']} | "
                         f"{arm['with']} | {arm['peak']:.1f} +- {arm['peak_sd']:.1f} | "
                         f"{arm['final']:.1f} +- {arm['final_sd']:.1f} | "
                         f"{arm['rejected'] if label == 'pruned' else '-'} |")
    lines.append(f"| Per-client GA | full | {ref['n']} | {', '.join(map(str, ref['drops']))} | {ref['total']} | "
                 f"{ref['with']} | {ref['peak']:.1f} +- {ref['peak_sd']:.1f} | {ref['final']:.1f} +- {ref['final_sd']:.1f} | - |")
    lines.append("")
    lines.append("Gap to per-client GA on the same seeds (pruned searcher minus per-client, pp):\n")
    lines.append("| Searcher | Peak | Final | Closed (within 1 pp on both) |")
    lines.append("|---|---:|---:|---|")
    for name, _, _ in SEARCHERS:
        a = pruned[name]
        if not a["n"]:
            lines.append(f"| {name} | - | - | no runs yet |")
            continue
        dp, df = a["peak"] - ref["peak"], a["final"] - ref["final"]
        closed = abs(dp) <= GAP_CLOSED_PP and abs(df) <= GAP_CLOSED_PP
        lines.append(f"| {name} | {dp:+.1f} | {df:+.1f} | {'yes' if closed else 'no'} |")
    text = "\n".join(lines) + "\n"
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
