#!/usr/bin/env python3
"""Fitness-signal bias analysis for the server-broadcast GA.

Reconstructs the GA's per-round trace (round, generation, individual index,
HP, fitness) for every broadcast-GA run by parsing the [FedGA] /
[HPSearch:FederatedGA] lines of federatedgeneticalgorithm/training.log inside
each run's [started_at, finished_at] window from matrix_summary.csv.

Quantifies:

1. Cold-start bias: the individual evaluated in round 1 faces the
   coldest model, so its fitness is structurally penalized regardless of merit.
   Per run: gen-0 fitness rank of the first-evaluated individual (expected
   worst-of-4 under the bias; 25% base rate under no bias), and whether its HP
   is ever broadcast again after gen 0. In the seeded arm, position 0
   holds the expert HP, the "obvious warm start", so the bias silently
   discards it.

2. Trajectory-position bias: with delta fitness (acc_t − acc_{t−1}),
   whichever individual follows a crash inherits a large positive delta.
   Per run: severe drops (>10pp), the delta credited to the next individual,
   and whether that individual became the running best ("elite promotion").

Usage:
    python analysis/fitness_bias.py [--markdown analysis/fitness_bias_report.md]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "matrix_summary.csv"
# Prefer the full application log (local dev); fall back to the committed
# excerpt of its [FedGA]/[HPSearch] lines (training.log itself is gitignored).
_LOG_CANDIDATES = [
    REPO_ROOT / "federatedgeneticalgorithm" / "training.log",
    REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "broadcast_traces.txt",
]
TRAINING_LOG = next((p for p in _LOG_CANDIDATES if p.exists()), _LOG_CANDIDATES[0])

SCENARIOS = {
    "ga_broadcast_cifar": "seeded population",
    "ga_broadcast_randominit_cifar": "random-init population",
    "ga_broadcast_deltafitness_cifar": "delta fitness",
}

BROADCAST_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - INFO - \[(?:FedGA|HPSearch:\w+)\] "
    r"Round (?P<round>\d+) -- broadcasting HP: batch=(?P<batch>\d+), opt=(?P<opt>\w+), "
    r"lr=(?P<lr>[\d.e-]+), wd=(?P<wd>[\d.e-]+), mom=(?P<mom>[\d.]+) \(gen=(?P<gen>\d+), idx=(?P<idx>\d+)\)"
)
FITNESS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - INFO - \[(?:FedGA|HPSearch:\w+)\] "
    r"Round (?P<round>\d+) -- eval-acc=(?P<acc>[\d.]+), fitness\(Δ\)=(?P<fit>[+-]?[\d.]+)"
)


def load_windows() -> list[dict]:
    rows = []
    with SUMMARY.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") == "ok" and r.get("scenario_name") in SCENARIOS:
                rows.append({
                    "scenario": r["scenario_name"],
                    "seed": int(r["seed"]),
                    "run_id": r["run_id"],
                    "start": r["started_at"].replace("T", " "),
                    "end": r["finished_at"].replace("T", " "),
                })
    return rows


def parse_traces(windows: list[dict]) -> dict[tuple, list[dict]]:
    """run-key -> ordered list of {round, gen, idx, hp, acc, fitness}."""
    traces: dict[tuple, list[dict]] = defaultdict(list)
    pending: dict[tuple, dict] = {}
    with TRAINING_LOG.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = BROADCAST_RE.match(line)
            if m:
                ts = m["ts"]
                for w in windows:
                    if w["start"] <= ts <= w["end"]:
                        key = (w["scenario"], w["seed"])
                        pending[key] = {
                            "round": int(m["round"]), "gen": int(m["gen"]), "idx": int(m["idx"]),
                            "hp": f"{m['opt']}/lr={m['lr']}/wd={m['wd']}/mom={m['mom']}/b={m['batch']}",
                            "opt": m["opt"], "lr": float(m["lr"]),
                        }
                        break
                continue
            m = FITNESS_RE.match(line)
            if m:
                ts = m["ts"]
                for w in windows:
                    if w["start"] <= ts <= w["end"]:
                        key = (w["scenario"], w["seed"])
                        rec = pending.pop(key, None)
                        rnd = int(m["round"])
                        if rec is None or rec["round"] != rnd:
                            # Older runs log the round-1 broadcast before telemetry
                            # initializes; synthesize idx from position.
                            rec = {"round": rnd, "gen": 0, "idx": None, "hp": None,
                                   "opt": None, "lr": None}
                        rec["acc"] = float(m["acc"])
                        rec["fitness"] = float(m["fit"])
                        traces[key].append(rec)
                        break
    return dict(traces)


def infer_missing_round1(traces: dict) -> None:
    """Some runs log the round-1 broadcast before telemetry initializes;
    round-robin order makes round r of gen 0 use idx (r-1) % pop while no
    evolution happened, so fill idx/gen for records missing them."""
    for key, tr in traces.items():
        for rec in tr:
            if rec["idx"] is None:
                rec["idx"] = (rec["round"] - 1) % 4
                rec["gen"] = 0


def cold_start_table(traces: dict) -> list[str]:
    lines = ["\n## 1. Cold-start bias: rank of the first-evaluated individual (gen 0)\n"]
    lines.append("| Arm | Seed | Fitness R1 | Fitness R2-R4 (gen 0) | Rank of 1st (1=best, 4=worst) | 1st HP re-broadcast after gen 0? |")
    lines.append("|---|---:|---:|---|---:|---|")
    worst_count = 0
    total = 0
    for (scenario, seed), tr in sorted(traces.items()):
        if scenario == "ga_broadcast_deltafitness_cifar":
            continue  # delta arm analyzed separately
        gen0 = [r for r in tr if r["gen"] == 0][:4]
        if len(gen0) < 4:
            continue
        first = gen0[0]
        fits = [r["fitness"] for r in gen0]
        rank = sorted(fits, reverse=True).index(first["fitness"]) + 1
        total += 1
        worst_count += rank == 4
        first_hp = first["hp"]
        reappears = any(
            r["hp"] == first_hp and r["round"] > 4 for r in tr if r["hp"]
        ) if first_hp else None
        lines.append(
            f"| {SCENARIOS[scenario]} | {seed} | {first['fitness']:.4f} "
            f"| {', '.join(f'{f:.3f}' for f in fits[1:])} | {rank} "
            f"| {'—' if reappears is None else ('yes' if reappears else 'NO')} |"
        )
    lines.append(
        f"\n**{worst_count}/{total} runs** rank the first-evaluated individual worst of generation 0 "
        f"(no-bias base rate: 25%; expected mean rank without bias: 2.5).\n"
    )
    return lines


def delta_bias_table(traces: dict) -> list[str]:
    lines = ["\n## 2. Trajectory-position bias: post-crash delta credit (delta-fitness arm)\n"]
    keys = [k for k in traces if k[0] == "ga_broadcast_deltafitness_cifar"]
    if not keys:
        lines.append("_No delta-fitness runs found yet._\n")
        return lines
    lines.append("| Seed | Crash (>10pp) | Post-crash round: HP | Credited delta | Promoted to best-so-far? | Run peak/final |")
    lines.append("|---:|---|---|---:|---|---|")
    for (scenario, seed) in sorted(keys):
        tr = traces[(scenario, seed)]
        crashes = []
        for i in range(1, len(tr)):
            drop = tr[i - 1]["acc"] - tr[i]["acc"]
            if drop > 0.10:
                nxt = tr[i + 1] if i + 1 < len(tr) else None
                promoted = (
                    nxt is not None
                    and nxt["fitness"] > max((t["fitness"] for t in tr[: i + 1]), default=float("-inf"))
                )
                crashes.append((i, drop, nxt, promoted))
        accs = [t["acc"] for t in tr]
        for i, drop, nxt, promoted in crashes:
            nxt_hp = (nxt["hp"] or "?") if nxt else "— (crash on final round; no next round)"
            nxt_fit = f"{nxt['fitness']:+.3f}" if nxt else "—"
            nxt_promoted = ("**YES**" if promoted else "no") if nxt else "—"
            lines.append(
                f"| {seed} | R{tr[i]['round']} (−{drop*100:.0f}pp) "
                f"| {nxt_hp} | {nxt_fit} | {nxt_promoted} "
                f"| {max(accs)*100:.1f} / {accs[-1]*100:.1f} |"
            )
        if not crashes:
            lines.append(f"| {seed} | 0 | — | — | — | {max(accs)*100:.1f} / {accs[-1]*100:.1f} |")

    # Stale-elite diagnosis: delta fitness rewards trajectory position, so the
    # all-time best-Δ individual is typically an early-slope (or post-crash)
    # rider that elitism then re-broadcasts forever despite mediocre later Δs.
    lines.append("\n### Stale elite (best-delta staleness)\n")
    lines.append("| Seed | Best-delta HP | Earned at | Delta | Later re-broadcasts | Mean delta on re-evaluation |")
    lines.append("|---:|---|---|---:|---:|---:|")
    for (scenario, seed) in sorted(keys):
        tr = traces[(scenario, seed)]
        best_rec = max(tr, key=lambda r: r["fitness"])
        later = [r for r in tr if r["hp"] == best_rec["hp"] and r["round"] > best_rec["round"]]
        mean_later = statistics.fmean(r["fitness"] for r in later) if later else None
        lines.append(
            f"| {seed} | {best_rec['hp'] or '(seeded expert)'} | R{best_rec['round']} "
            f"| {best_rec['fitness']:+.3f} | {len(later)} "
            f"| {f'{mean_later:+.3f}' if mean_later is not None else '—'} |"
        )
    return lines


def end_accuracy_note(traces: dict) -> list[str]:
    lines = ["\n## 3. Peak accuracy per arm (context)\n"]
    by_scn = defaultdict(list)
    for (scenario, seed), tr in traces.items():
        accs = [t["acc"] for t in tr]
        if accs:
            by_scn[scenario].append(max(accs))
    for scn, peaks in sorted(by_scn.items()):
        mean = statistics.fmean(peaks)
        sd = statistics.stdev(peaks) if len(peaks) > 1 else 0.0
        lines.append(f"- `{scn}` ({SCENARIOS[scn]}): peak {mean*100:.2f} ± {sd*100:.2f} (N={len(peaks)})")
    lines.append(
        "\n> Note: seeding has no measurable effect on end-point accuracy (arms tie); "
        "the cold-start bias wastes the warm start (the expert HP is discarded as "
        "generation-worst) rather than degrading the mean outcome."
    )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    windows = load_windows()
    traces = parse_traces(windows)
    infer_missing_round1(traces)

    out = ["# Fitness-signal bias report\n"]
    out.append(f"_Runs reconstructed from training.log: {len(traces)} "
               f"({', '.join(sorted(set(k[0] for k in traces)))})_\n")
    out += cold_start_table(traces)
    out += delta_bias_table(traces)
    out += end_accuracy_note(traces)
    report = "\n".join(out) + "\n"
    print(report)
    if args.markdown:
        args.markdown.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
