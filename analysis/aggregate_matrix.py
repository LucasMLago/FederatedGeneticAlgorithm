#!/usr/bin/env python3
"""Aggregate a Phase-2 matrix sweep CSV into a per-scenario summary table.

Reads the summary CSV that `scripts/run_matrix.py` appends to (one row per
completed `(config, seed)` run) and emits a Markdown table with mean / sd /
CI95% / N for peak and final eval accuracy, grouped by scenario.

Designed to be safe to run while a sweep is still in progress; partial
results show up immediately so we can sanity-check seed 0 without waiting
for seeds 1..4 to finish.

Usage:
    python analysis/aggregate_matrix.py
    python analysis/aggregate_matrix.py --summary path/to/matrix_summary.csv
    python analysis/aggregate_matrix.py --markdown report.md
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "matrix_summary.csv"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Path to the matrix summary CSV (default: {DEFAULT_SUMMARY}).",
    )
    p.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="If given, also write the report to this Markdown file.",
    )
    p.add_argument(
        "--only-ok",
        action="store_true",
        default=True,
        help="Skip rows with status != 'ok' (default: on).",
    )
    p.add_argument(
        "--include-failed",
        action="store_false",
        dest="only_ok",
        help="Include failed rows in the table (useful for diagnosing).",
    )
    return p.parse_args()


def _ci95_halfwidth(values: List[float]) -> float | None:
    """Return the 95% normal-approx CI half-width, or None if N < 2.

    For Phase 2 the per-cell N is 3-5, so the normal approximation is rough;
    the table reports it for orientation but the paper-ready stats will use
    Mann-Whitney U.
    """
    n = len(values)
    if n < 2:
        return None
    sd = statistics.stdev(values)
    return 1.96 * sd / math.sqrt(n)


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{digits}f}"


def _load_rows(path: Path, only_ok: bool) -> List[Dict[str, str]]:
    if not path.exists():
        print(f"[aggregate] summary CSV not found: {path}", file=sys.stderr)
        return []
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if only_ok:
        rows = [r for r in rows if r.get("status") == "ok"]
    return rows


def _group_by_scenario(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        scenario = r.get("scenario_name") or r.get("config", "?")
        groups[scenario].append(r)
    return groups


def _stats_for(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "sd": None, "ci95": None, "min": None, "max": None, "n": 0}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else None
    return {
        "mean": mean,
        "sd": sd,
        "ci95": _ci95_halfwidth(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _build_report(groups: Dict[str, List[Dict[str, str]]]) -> str:
    lines: List[str] = []
    lines.append("# Matrix summary\n")

    if not groups:
        lines.append("_No successful runs found in summary CSV._\n")
        return "\n".join(lines)

    lines.append("| Scenario | N | Peak mean (%) | Peak ±sd | Peak 95% CI ± | Final mean (%) | Final ±sd | Final 95% CI ± | Wall avg (min) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for scenario in sorted(groups):
        rows = groups[scenario]
        peaks = [float(r["peak_eval_acc"]) for r in rows if r.get("peak_eval_acc")]
        finals = [float(r["final_eval_acc"]) for r in rows if r.get("final_eval_acc")]
        walls = [float(r["wall_seconds"]) for r in rows if r.get("wall_seconds")]
        ps = _stats_for(peaks)
        fs = _stats_for(finals)
        wall_avg = (sum(walls) / len(walls) / 60.0) if walls else None
        lines.append(
            "| `{name}` | {n} | {pm} | {ps} | {pc} | {fm} | {fs} | {fc} | {wm} |".format(
                name=scenario,
                n=ps["n"],
                pm=_fmt_pct(ps["mean"]),
                ps=_fmt_pct(ps["sd"]),
                pc=_fmt_pct(ps["ci95"]),
                fm=_fmt_pct(fs["mean"]),
                fs=_fmt_pct(fs["sd"]),
                fc=_fmt_pct(fs["ci95"]),
                wm=f"{wall_avg:.1f}" if wall_avg is not None else "—",
            )
        )

    lines.append("\n## Per-seed detail\n")
    lines.append("| Scenario | Seed | Peak (%) | Final (%) | Wall (min) | run_id | git_sha[:7] |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for scenario in sorted(groups):
        for row in sorted(groups[scenario], key=lambda r: int(r.get("seed") or 0)):
            peak = row.get("peak_eval_acc")
            final = row.get("final_eval_acc")
            wall = row.get("wall_seconds")
            sha = (row.get("git_sha") or "")[:7] or "—"
            lines.append(
                "| `{name}` | {seed} | {pk} | {fn} | {wl} | `{rid}` | `{sha}` |".format(
                    name=scenario,
                    seed=row.get("seed", "?"),
                    pk=_fmt_pct(float(peak)) if peak else "—",
                    fn=_fmt_pct(float(final)) if final else "—",
                    wl=f"{float(wall) / 60.0:.1f}" if wall else "—",
                    rid=row.get("run_id", ""),
                    sha=sha,
                )
            )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    rows = _load_rows(args.summary, args.only_ok)
    groups = _group_by_scenario(rows)
    report = _build_report(groups)

    print(report)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
        print(f"\n[aggregate] wrote {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
