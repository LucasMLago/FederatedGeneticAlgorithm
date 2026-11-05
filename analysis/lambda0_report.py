#!/usr/bin/env python3
"""Per-client GA without the drift penalty (lambda = 0) against the standard per-client GA, seeds 0-2.

Reviewer question: does the drift term in the fitness (`acc_val - lambda * d`) explain why per-client
search is stable? The lambda-0 runs zero it out. The script reports peak and final accuracy, severe
drops (more than 10 pp in one round) and the share of client updates that trained a harmful
configuration, counted as in client_agreement.py.

Usage:
    python analysis/lambda0_report.py
    python analysis/lambda0_report.py --markdown analysis/lambda0_report.md
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import client_agreement as ca

REPO_ROOT = ca.REPO_ROOT
FINAL = ca.SUMMARY
PROBES = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "matrix_summary_probes.csv"
BASELINE = "ga_perclient_cifar"
LAMBDA0 = "ga_perclient_lambda0_cifar"
SEEDS = range(3)


def rows(path: Path, scenario: str, seeds=None) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return sorted((r for r in csv.DictReader(fh) if r["scenario_name"] == scenario and r["status"] == "ok"
                       and (seeds is None or int(r["seed"]) in seeds)), key=lambda r: int(r["seed"]))


def per_run(rs: list[dict]) -> list[dict]:
    out = []
    for r in rs:
        rid = r["run_id"]
        cfgs = [c for round_cfgs in ca.run_rounds(rid).values() for c in round_cfgs]
        out.append({
            "seed": int(r["seed"]), "peak": 100 * float(r["peak_eval_acc"]), "final": 100 * float(r["final_eval_acc"]),
            "drops": len(ca.drops(ca.eval_acc(rid))), "updates": len(cfgs), "harmful": sum(ca.harmful(c) for c in cfgs),
        })
    return out


def share(runs: list[dict]) -> float:
    return 100 * sum(r["harmful"] for r in runs) / sum(r["updates"] for r in runs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    arms = {
        "per-client GA, seeds 0-2": per_run(rows(FINAL, BASELINE, SEEDS)),
        "per-client GA, all ten seeds": per_run(rows(FINAL, BASELINE)),
        "per-client GA, lambda = 0, seeds 0-2": per_run(rows(PROBES, LAMBDA0, SEEDS)),
    }
    lines = ["# Per-client GA without the drift penalty", ""]
    lines.append(f"_Source: `{FINAL.relative_to(REPO_ROOT)}` and `{PROBES.relative_to(REPO_ROOT)}`, with the per-run "
                 f"telemetry. Severe drop: more than {ca.SEVERE_DROP:.0f} pp in one round. Harmful: lion with lr >= 5e-3, "
                 f"or adam, adamw, radam with lr = 1e-2._\n")
    lines.append("| Arm | Runs | Peak % | Final % | Severe drops (runs with one) | Harmful updates |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, runs in arms.items():
        peaks, finals = [r["peak"] for r in runs], [r["final"] for r in runs]
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else float("nan")
        lines.append(f"| {name} | {len(runs)} | {statistics.fmean(peaks):.2f} +- {sd(peaks):.2f} | "
                     f"{statistics.fmean(finals):.2f} +- {sd(finals):.2f} | {sum(r['drops'] for r in runs)} "
                     f"({sum(r['drops'] > 0 for r in runs)}/{len(runs)}) | "
                     f"{share(runs):.1f}% ({sum(r['harmful'] for r in runs)}/{sum(r['updates'] for r in runs)}) |")
    lines.append("")
    lines.append("Lambda-0 runs, one by one:\n")
    lines.append("| Seed | Peak % | Final % | Severe drops | Harmful updates |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in arms["per-client GA, lambda = 0, seeds 0-2"]:
        lines.append(f"| {r['seed']} | {r['peak']:.2f} | {r['final']:.2f} | {r['drops']} | {r['harmful']}/{r['updates']} |")
    text = "\n".join(lines) + "\n"
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
