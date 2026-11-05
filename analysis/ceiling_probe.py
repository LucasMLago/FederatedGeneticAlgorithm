#!/usr/bin/env python3
"""Does a fixed SGD configuration next to the expert beat it?

The expert (sgd, lr 0.01, momentum 0.9, batch 64, no weight decay) stayed above every search in the
main scenario. Eight fixed variants, each changing one or two values of the expert, test whether
something nearby does better: five stay inside the GA's grid and three go just outside it. One run
each, seed 0 (`scripts/make_probe_configs.py` writes the configs).

The reference is the expert on seed 0 and the seed-to-seed sd of its peak over the paper's runs.

Usage:
    python analysis/ceiling_probe.py
    python analysis/ceiling_probe.py --markdown analysis/ceiling_probe_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts"
SUMMARY = Path(os.environ.get("FGA_SUMMARY", ARTIFACTS / "matrix_summary_final.csv"))
PROBES = ARTIFACTS / "matrix_summary_probes.csv"
RUNS_DIR = ARTIFACTS / "runs"

EXPERT = {"DEFAULT_LR": 0.01, "DEFAULT_WEIGHT_DECAY": 0.0, "DEFAULT_MOMENTUM": 0.9, "DEFAULT_BATCH_SIZE": 64}
GRID = {
    "DEFAULT_LR": [5e-4, 1e-3, 3e-3, 5e-3, 1e-2],
    "DEFAULT_WEIGHT_DECAY": [0.0, 1e-5, 1e-4, 1e-3],
    "DEFAULT_MOMENTUM": [0.5, 0.7, 0.9, 0.95],
    "DEFAULT_BATCH_SIZE": [64, 128],
}
SHORT = {"DEFAULT_LR": "lr", "DEFAULT_WEIGHT_DECAY": "wd", "DEFAULT_MOMENTUM": "momentum", "DEFAULT_BATCH_SIZE": "batch"}


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["status"] == "ok"]


def in_grid(value, grid) -> bool:
    return any(math.isclose(float(value), g, rel_tol=1e-9, abs_tol=1e-12) for g in grid)


def probe_hp(run_id: str) -> dict:
    return json.loads((RUNS_DIR / run_id / "resolved_config.json").read_text(encoding="utf-8"))


def fmt(value) -> str:
    return f"{value:g}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    expert_rows = [r for r in rows(SUMMARY) if r["scenario_name"] == "fixed_expert_cifar"]
    seed0 = next(r for r in expert_rows if int(r["seed"]) == 0)
    expert_peak = 100 * float(seed0["peak_eval_acc"])
    expert_final = 100 * float(seed0["final_eval_acc"])
    expert_sd = statistics.stdev(100 * float(r["peak_eval_acc"]) for r in expert_rows)

    probes = []
    for r in rows(PROBES):
        if not r["scenario_name"].startswith("probe_sgd"):
            continue
        hp = probe_hp(r["run_id"])
        changed = {k: hp[k] for k in EXPERT if not math.isclose(float(hp[k]), EXPERT[k])}
        probes.append({
            "name": r["scenario_name"], "run_id": r["run_id"],
            "change": ", ".join(f"{SHORT[k]} {fmt(v)}" for k, v in changed.items()),
            "inside": all(in_grid(hp[k], GRID[k]) for k in GRID),
            "peak": 100 * float(r["peak_eval_acc"]), "final": 100 * float(r["final_eval_acc"]),
            "wall": float(r["wall_seconds"]) / 60,
        })
    probes.sort(key=lambda p: -p["peak"])
    best = probes[0]
    gain = best["peak"] - expert_peak

    lines = ["# Ceiling probe: fixed SGD variants of the expert", ""]
    lines.append(f"_Source: `{PROBES.relative_to(REPO_ROOT)}` for the probes, `{SUMMARY.relative_to(REPO_ROOT)}` for "
                 f"the expert. Seed 0, one run per probe._\n")
    verdict = "within" if gain <= expert_sd else "above"
    lines.append(f"**The best probe is `{best['name']}` ({best['change']}), peak {best['peak']:.2f}%, "
                 f"{gain:+.2f} pp against the expert's {expert_peak:.2f}% on seed 0, which is {verdict} the "
                 f"expert's seed-to-seed sd ({expert_sd:.2f} pp over {len(expert_rows)} seeds).** "
                 f"Probe peaks range from {min(p['peak'] for p in probes):.2f} to {best['peak']:.2f}%; "
                 f"{sum(p['inside'] for p in probes)} probes are inside the grid and "
                 f"{sum(not p['inside'] for p in probes)} outside it.\n")
    lines.append("| Probe | Change from the expert | In the grid | Peak % | Final % | Wall (min) | run_id |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    lines.append(f"| expert, seed 0 | none | yes | {expert_peak:.2f} | {expert_final:.2f} | "
                 f"{float(seed0['wall_seconds']) / 60:.1f} | `{seed0['run_id']}` |")
    for p in probes:
        lines.append(f"| `{p['name']}` | {p['change']} | {'yes' if p['inside'] else 'no'} | {p['peak']:.2f} | "
                     f"{p['final']:.2f} | {p['wall']:.1f} | `{p['run_id']}` |")
    text = "\n".join(lines) + "\n"
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
