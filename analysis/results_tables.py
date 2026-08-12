#!/usr/bin/env python3
"""Builds the results tables: one report with every table the paper needs.

Consolidates the matrix summary (scalar per-run metrics) with the per-round
server telemetry (eval curves) into:

  1. Per-scenario summary stats (peak / final mean ± sd ± CI95, wall minutes)
     grouped by experiment family (CIFAR-main, FEMNIST, CIFAR-small, α=0.1,
     failure-mode arms).
  2. Round-level robustness metrics per scenario: severe single-round drops
     (Δ eval-acc < −10pp), max drop, terminal-collapse detection (metrics stop
     parsing before the final round), rounds-to-{70,75,80}%.
  3. Pairwise statistical tests for the comparisons reported in the paper:
     Mann-Whitney U (unpaired) + Wilcoxon signed-rank (paired by seed) on peak
     and final accuracy.

Usage:
    python analysis/results_tables.py
    python analysis/results_tables.py --markdown analysis/results_tables.md
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import mannwhitneyu, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "matrix_summary.csv"
RUNS_DIR = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "runs"

SEVERE_DROP_PP = 0.10  # Δ eval-acc < −10pp in one round

# Scenario → (family, coupling label) so tables group meaningfully.
FAMILIES = {
    "fixed_expert_cifar": ("cifar-main", "fixed HP (expert)"),
    "fixed_naive_cifar": ("cifar-main", "fixed HP (naive)"),
    "ga_perclient_cifar": ("cifar-main", "GA zero-coupling"),
    "ga_surrogate_cifar": ("cifar-main", "GA medium-coupling"),
    "ga_broadcast_cifar": ("cifar-main", "GA high-coupling"),
    "rs_broadcast_cifar": ("cifar-main", "RS high-coupling"),
    "tpe_broadcast_cifar": ("cifar-main", "TPE high-coupling"),
    "fixed_expert_femnist": ("femnist", "fixed HP (expert)"),
    "fixed_naive_femnist": ("femnist", "fixed HP (naive)"),
    "ga_perclient_femnist": ("femnist", "GA zero-coupling"),
    "ga_surrogate_femnist": ("femnist", "GA medium-coupling"),
    "ga_broadcast_femnist": ("femnist", "GA high-coupling"),
    "rs_broadcast_femnist": ("femnist", "RS high-coupling"),
    "tpe_broadcast_femnist": ("femnist", "TPE high-coupling"),
    "fixed_expert_smallcnn": ("cifar-small", "fixed HP (expert)"),
    "fixed_naive_smallcnn": ("cifar-small", "fixed HP (naive)"),
    "ga_perclient_smallcnn": ("cifar-small", "GA zero-coupling"),
    "ga_surrogate_smallcnn": ("cifar-small", "GA medium-coupling"),
    "ga_broadcast_smallcnn": ("cifar-small", "GA high-coupling"),
    "fixed_expert_alpha01": ("cifar-alpha01", "fixed HP (expert)"),
    "fixed_naive_alpha01": ("cifar-alpha01", "fixed HP (naive)"),
    "ga_perclient_alpha01": ("cifar-alpha01", "GA zero-coupling"),
    "ga_surrogate_alpha01": ("cifar-alpha01", "GA medium-coupling"),
    "ga_broadcast_alpha01": ("cifar-alpha01", "GA high-coupling"),
    "ga_broadcast_randominit_cifar": ("failure-modes", "broadcast GA, random-init pop"),
    "ga_broadcast_deltafitness_cifar": ("failure-modes", "broadcast GA, delta fitness"),
}

# (label, scenario_a, scenario_b): a vs b on peak/final. Paired by seed when
# the seed sets intersect; Mann-Whitney reported always.
COMPARISONS = [
    ("CIFAR: per-client vs FedGA", "ga_perclient_cifar", "ga_broadcast_cifar"),
    ("CIFAR: per-client vs surrogate", "ga_perclient_cifar", "ga_surrogate_cifar"),
    ("CIFAR: surrogate vs FedGA", "ga_surrogate_cifar", "ga_broadcast_cifar"),
    ("CIFAR: FedGA vs RS (broadcast family)", "ga_broadcast_cifar", "rs_broadcast_cifar"),
    ("CIFAR: FedGA vs TPE (broadcast family)", "ga_broadcast_cifar", "tpe_broadcast_cifar"),
    ("CIFAR: per-client vs naive baseline", "ga_perclient_cifar", "fixed_naive_cifar"),
    ("CIFAR: FedGA vs naive baseline", "ga_broadcast_cifar", "fixed_naive_cifar"),
    ("CIFAR: expert vs per-client", "fixed_expert_cifar", "ga_perclient_cifar"),
    ("Broadcast GA: seeded vs random-init population", "ga_broadcast_cifar", "ga_broadcast_randominit_cifar"),
    ("Broadcast GA: absolute vs delta fitness", "ga_broadcast_cifar", "ga_broadcast_deltafitness_cifar"),
    ("FEMNIST: per-client vs FedGA", "ga_perclient_femnist", "ga_broadcast_femnist"),
    ("FEMNIST: expert vs naive", "fixed_expert_femnist", "fixed_naive_femnist"),
    ("FEMNIST: expert vs FedGA", "fixed_expert_femnist", "ga_broadcast_femnist"),
    ("Small: expert vs per-client GA", "fixed_expert_smallcnn", "ga_perclient_smallcnn"),
    ("Small: expert vs FedGA", "fixed_expert_smallcnn", "ga_broadcast_smallcnn"),
    ("Small: per-client vs FedGA", "ga_perclient_smallcnn", "ga_broadcast_smallcnn"),
    ("Small: per-client vs surrogate", "ga_perclient_smallcnn", "ga_surrogate_smallcnn"),
    ("α=0.1: expert vs per-client GA", "fixed_expert_alpha01", "ga_perclient_alpha01"),
    ("α=0.1: expert vs FedGA", "fixed_expert_alpha01", "ga_broadcast_alpha01"),
    ("α=0.1: per-client vs FedGA", "ga_perclient_alpha01", "ga_broadcast_alpha01"),
    ("α=0.1: per-client vs surrogate", "ga_perclient_alpha01", "ga_surrogate_alpha01"),
    ("α=0.1: surrogate vs FedGA", "ga_surrogate_alpha01", "ga_broadcast_alpha01"),
    ("α=0.1: expert vs naive", "fixed_expert_alpha01", "fixed_naive_alpha01"),
    ("α=0.1: naive vs FedGA", "fixed_naive_alpha01", "ga_broadcast_alpha01"),
]


def load_summary() -> list[dict]:
    with SUMMARY.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]
    for r in rows:
        r["seed"] = int(r["seed"])
        r["peak"] = float(r["peak_eval_acc"]) if r.get("peak_eval_acc") else None
        r["final"] = float(r["final_eval_acc"]) if r.get("final_eval_acc") else None
        r["wall_min"] = float(r["wall_seconds"]) / 60.0 if r.get("wall_seconds") else None
    return rows


def eval_curve(run_id: str) -> list[float | None]:
    """Per-round eval-acc; None where the row exists but the metric is empty."""
    path = RUNS_DIR / run_id / "server_aggregated_rounds.csv"
    if not path.exists():
        return []
    by_round: dict[int, float | None] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("phase") != "evaluate":
                continue
            rnd = int(row["server_round"])
            raw = row.get("eval-acc") or ""
            try:
                by_round[rnd] = float(raw)
            except ValueError:
                by_round[rnd] = None
    if not by_round:
        return []
    return [by_round.get(i) for i in range(1, max(by_round) + 1)]


def robustness(curve: list[float | None]) -> dict:
    vals = [(i, v) for i, v in enumerate(curve) if v is not None]
    out = {
        "n_rounds_parsed": len(vals),
        "n_rounds_total": len(curve),
        "terminal_collapse": bool(curve) and curve[-1] is None,
        "severe_drops": 0,
        "max_drop_pp": 0.0,
        "worst_drop_round": None,
        "rounds_to_70": None,
        "rounds_to_75": None,
        "rounds_to_80": None,
    }
    prev = None
    for i, v in vals:
        if prev is not None:
            drop = prev - v
            if drop > SEVERE_DROP_PP:
                out["severe_drops"] += 1
            if drop > out["max_drop_pp"]:
                out["max_drop_pp"] = drop
                out["worst_drop_round"] = i + 1
        prev = v
        for thr, key in ((0.70, "rounds_to_70"), (0.75, "rounds_to_75"), (0.80, "rounds_to_80")):
            if out[key] is None and v >= thr:
                out[key] = i + 1
    return out


def stats_block(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values) if values else None
    sd = statistics.stdev(values) if n >= 2 else None
    ci = 1.96 * sd / math.sqrt(n) if sd is not None else None
    return {"n": n, "mean": mean, "sd": sd, "ci95": ci}


def fmt(x, digits=2):
    return "—" if x is None else f"{x * 100:.{digits}f}"


def pairwise(rows_by_scn: dict, a: str, b: str, metric: str):
    ra = {r["seed"]: r[metric] for r in rows_by_scn.get(a, []) if r[metric] is not None}
    rb = {r["seed"]: r[metric] for r in rows_by_scn.get(b, []) if r[metric] is not None}
    if not ra or not rb:
        return None
    va, vb = list(ra.values()), list(rb.values())
    delta = statistics.fmean(va) - statistics.fmean(vb)
    try:
        mw_p = mannwhitneyu(va, vb, alternative="two-sided").pvalue
    except ValueError:
        mw_p = float("nan")
    shared = sorted(set(ra) & set(rb))
    wil_p, n_paired = None, len(shared)
    if n_paired >= 3:
        diffs = [ra[s] - rb[s] for s in shared]
        if any(abs(d) > 1e-12 for d in diffs):
            try:
                wil_p = wilcoxon(diffs).pvalue
            except ValueError:
                wil_p = None
    return {"delta_pp": delta * 100, "mw_p": mw_p, "wil_p": wil_p,
            "n_a": len(va), "n_b": len(vb), "n_paired": n_paired}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    rows = load_summary()
    rows_by_scn: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rows_by_scn[r["scenario_name"]].append(r)

    # Round-level metrics per run
    for r in rows:
        r["rob"] = robustness(eval_curve(r["run_id"]))

    lines: list[str] = []
    lines.append("# Results tables\n")
    lines.append(f"_Fonte: `{SUMMARY.relative_to(REPO_ROOT)}` + telemetria por round em `artifacts/runs/`. "
                 f"Runs ok: {len(rows)}._\n")

    # ---- 1. summary tables by family ----
    fam_order = ["cifar-main", "femnist", "cifar-small", "cifar-alpha01", "failure-modes"]
    fam_titles = {
        "cifar-main": "CIFAR-10 / ResNet 11M / α=0.5 (cenário principal)",
        "femnist": "FEMNIST / CNN LEAF / α=0.5",
        "cifar-small": "CIFAR-10 / SmallCNN 530K / α=0.5",
        "cifar-alpha01": "CIFAR-10 / ResNet 11M / α=0.1 (boundary)",
        "failure-modes": "Braços de failure-mode do fitness signal (α=0.5)",
    }
    for fam in fam_order:
        scns = [s for s, (f, _) in FAMILIES.items() if f == fam and s in rows_by_scn]
        if not scns:
            continue
        lines.append(f"\n## {fam_titles[fam]}\n")
        lines.append("| Cenário | Regime | N | Peak % (±sd) | Final % (±sd) | Wall (min) | Drops>10pp (runs c/ drop) | Colapso terminal | R→70 / 75 / 80 (mediana) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for scn in scns:
            rs = sorted(rows_by_scn[scn], key=lambda r: r["seed"])
            ps = stats_block([r["peak"] for r in rs if r["peak"] is not None])
            fs = stats_block([r["final"] for r in rs if r["final"] is not None])
            wall = statistics.fmean([r["wall_min"] for r in rs if r["wall_min"]])
            drops_total = sum(r["rob"]["severe_drops"] for r in rs)
            runs_with_drop = sum(1 for r in rs if r["rob"]["severe_drops"] > 0)
            collapses = sum(1 for r in rs if r["rob"]["terminal_collapse"])
            med = {}
            for key in ("rounds_to_70", "rounds_to_75", "rounds_to_80"):
                vals = [r["rob"][key] for r in rs if r["rob"][key] is not None]
                med[key] = statistics.median(vals) if vals else None
            r2s = " / ".join("—" if med[k] is None else f"{med[k]:.0f}"
                             for k in ("rounds_to_70", "rounds_to_75", "rounds_to_80"))
            lines.append(
                f"| `{scn}` | {FAMILIES[scn][1]} | {ps['n']} "
                f"| {fmt(ps['mean'])} ± {fmt(ps['sd'])} "
                f"| {fmt(fs['mean'])} ± {fmt(fs['sd'])} "
                f"| {wall:.1f} | {drops_total} ({runs_with_drop}/{ps['n']}) "
                f"| {collapses}/{ps['n']} | {r2s} |"
            )

    # ---- 2. per-run drops detail (only runs with events) ----
    lines.append("\n## Runs com eventos de robustez (drop severo ou colapso)\n")
    lines.append("| Cenário | Seed | run_id | Drops | Max drop (pp) | Round do pior drop | Rounds parseados/total | Colapso |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: (r["scenario_name"], r["seed"])):
        rb = r["rob"]
        if rb["severe_drops"] == 0 and not rb["terminal_collapse"]:
            continue
        lines.append(
            f"| `{r['scenario_name']}` | {r['seed']} | `{r['run_id']}` | {rb['severe_drops']} "
            f"| {rb['max_drop_pp'] * 100:.1f} | {rb['worst_drop_round'] or '—'} "
            f"| {rb['n_rounds_parsed']}/{rb['n_rounds_total']} "
            f"| {'SIM' if rb['terminal_collapse'] else 'não'} |"
        )

    # ---- 3. pairwise tests ----
    lines.append("\n## Testes pareados (peak e final)\n")
    lines.append("| Comparação | Δpeak (pp, a−b) | MW-U p | Wilcoxon p (N pareado) | Δfinal (pp) | MW-U p | Wilcoxon p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, a, b in COMPARISONS:
        pk = pairwise(rows_by_scn, a, b, "peak")
        fn = pairwise(rows_by_scn, a, b, "final")
        if pk is None:
            lines.append(f"| {label} | _dados ausentes_ | | | | | |")
            continue
        wil_pk = f"{pk['wil_p']:.3f} (N={pk['n_paired']})" if pk["wil_p"] is not None else f"— (N={pk['n_paired']})"
        wil_fn = f"{fn['wil_p']:.3f}" if fn and fn["wil_p"] is not None else "—"
        fn_delta = f"{fn['delta_pp']:+.2f}" if fn else "—"
        fn_mw = f"{fn['mw_p']:.3f}" if fn else "—"
        lines.append(
            f"| {label} | {pk['delta_pp']:+.2f} | {pk['mw_p']:.3f} | {wil_pk} "
            f"| {fn_delta} | {fn_mw} | {wil_fn} |"
        )

    report = "\n".join(lines) + "\n"
    print(report)
    if args.markdown:
        args.markdown.write_text(report, encoding="utf-8")
        print(f"[master] wrote {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
