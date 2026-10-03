#!/usr/bin/env python3
"""Agreement between client configurations and the configurations behind severe drops.

Every client row is mapped to its server round (the first train-phase aggregation at or after the
row's timestamp). Per round, the script compares the configurations of all participating pairs and
checks whether any participant trained a harmful configuration (lion with lr >= 5e-3, or
adam/adamw/radam with lr = 1e-2).

Usage:
    python analysis/client_agreement.py
    python analysis/client_agreement.py --markdown analysis/client_agreement_report.md
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import statistics
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# FGA_SUMMARY picks another summary file, e.g. matrix_summary_final.csv
SUMMARY = Path(os.environ.get("FGA_SUMMARY", REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "matrix_summary.csv"))
RUNS_DIR = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "runs"

CLIENT_LEVEL = ["ga_perclient_cifar", "ga_surrogate_cifar", "ga_surrogate_nopool_cifar", "ga_perclient_cifar_r40",
                "ga_perclient_alpha01", "ga_surrogate_alpha01", "fedex_cifar", "ga_surrogate_longeval_cifar"]
BROADCAST = ["ga_broadcast_cifar", "ga_broadcast_noelite_cifar", "ga_broadcast_cifar_r40", "tpe_broadcast_cifar",
             "rs_broadcast_cifar"]
FIRST_ROUND = 5  # agreement skips generation 0 / the first participations
SEVERE_DROP = 10.0  # pp in one round


def ts(value: str) -> float:
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return float(value)


def config(r: dict) -> tuple:
    momentum = r["momentum"] if r["optimizer"] == "sgd" else "-"  # momentum is inert outside sgd
    return (r["batch_size"], r["optimizer"], float(r["lr"]), float(r["weight_decay"]), momentum)


def harmful(c: tuple) -> bool:
    return (c[1] == "lion" and c[2] >= 5e-3) or (c[1] in ("adam", "adamw", "radam") and c[2] >= 1e-2)


def run_rounds(run_id: str) -> dict[int, list[tuple]]:
    d = RUNS_DIR / run_id
    with (d / "server_aggregated_rounds.csv").open(encoding="utf-8") as fh:
        train_ts = sorted((ts(r["timestamp"]), int(r["server_round"])) for r in csv.DictReader(fh)
                          if r.get("phase") == "train")
    out: dict[int, list[tuple]] = {}
    with (d / "client_round_metrics.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t = ts(r["timestamp"])
            rnd = next((k for tt, k in train_ts if tt >= t), None)
            if rnd is not None:
                out.setdefault(rnd, []).append(config(r))
    return out


def eval_acc(run_id: str) -> dict[int, float]:
    with (RUNS_DIR / run_id / "server_aggregated_rounds.csv").open(encoding="utf-8") as fh:
        return {int(r["server_round"]): 100 * float(r["eval-acc"]) for r in csv.DictReader(fh) if r.get("eval-acc")}


def run_ids(scenario: str, seeds: range | None = None) -> list[str]:
    with SUMMARY.open(encoding="utf-8") as fh:
        return [r["run_id"] for r in csv.DictReader(fh) if r["scenario_name"] == scenario and r["status"] == "ok"
                and (seeds is None or int(r["seed"]) in seeds)]


def drops(acc: dict[int, float]) -> list[int]:
    return [k for k in sorted(acc) if k - 1 in acc and acc[k - 1] - acc[k] > SEVERE_DROP]


def client_level(scenario: str) -> dict | None:
    ids = run_ids(scenario)
    if not ids:
        return None
    same_cfg, same_opt, updates, harmful_updates = [], [], 0, 0
    drop_rounds = drop_harmful = other_rounds = other_harmful = 0
    for rid in ids:
        rounds, acc = run_rounds(rid), eval_acc(rid)
        dropped = set(drops(acc))
        for rnd, cfgs in rounds.items():
            updates += len(cfgs)
            harmful_updates += sum(harmful(c) for c in cfgs)
            any_harmful = any(harmful(c) for c in cfgs)
            if rnd in dropped:
                drop_rounds += 1
                drop_harmful += any_harmful
            elif rnd - 1 in acc:
                other_rounds += 1
                other_harmful += any_harmful
            pairs = list(itertools.combinations(cfgs, 2))
            if rnd >= FIRST_ROUND and pairs:
                same_cfg.append(sum(a == b for a, b in pairs) / len(pairs))
                same_opt.append(sum(a[1] == b[1] for a, b in pairs) / len(pairs))
    return {"runs": len(ids), "same_cfg": statistics.fmean(same_cfg), "same_opt": statistics.fmean(same_opt),
            "harmful_updates": harmful_updates / updates, "drops": drop_rounds, "drops_harmful": drop_harmful,
            "other_harmful": other_harmful / max(other_rounds, 1)}


def broadcast(scenario: str, seeds: range | None = None) -> dict | None:
    ids = run_ids(scenario, seeds)
    if not ids:
        return None
    n_rounds = n_harmful = n_drops = drops_harmful = 0
    recovery: list[int | None] = []
    for rid in ids:
        rounds, acc = run_rounds(rid), eval_acc(rid)
        for rnd, cfgs in rounds.items():
            n_rounds += 1
            n_harmful += harmful(cfgs[0])  # the same configuration for every participant
        for rnd in drops(acc):
            n_drops += 1
            drops_harmful += harmful(rounds[rnd][0]) if rnd in rounds else 0
            back = [k for k in sorted(acc) if k > rnd and acc[k] >= acc[rnd - 1] - 5]
            recovery.append(back[0] - rnd if back else None)
    return {"runs": len(ids), "rounds": n_rounds, "harmful": n_harmful, "drops": n_drops,
            "drops_harmful": drops_harmful, "recovery": recovery}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    lines = ["# Client agreement and harmful configurations\n",
             f"_Fonte: `{SUMMARY.relative_to(REPO_ROOT)}` + `client_round_metrics.csv` de cada run. "
             f"Nocivas: lion com lr >= 5e-3, adam/adamw/radam com lr = 1e-2._\n",
             "\n## Regimes em que cada cliente escolhe\n",
             f"| Cenário | Runs | Pares idênticos (round >= {FIRST_ROUND}) | Pares com mesmo otimizador "
             "| Atualizações nocivas | Quedas com cliente nocivo | Rounds sem queda com cliente nocivo |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for scn in CLIENT_LEVEL:
        r = client_level(scn)
        if r:
            lines.append(f"| `{scn}` | {r['runs']} | {100 * r['same_cfg']:.0f}% | {100 * r['same_opt']:.0f}% "
                         f"| {100 * r['harmful_updates']:.1f}% | {r['drops_harmful']}/{r['drops']} "
                         f"| {100 * r['other_harmful']:.0f}% |")
    lines += ["\n## Busca por broadcast (uma configuração por round para todos)\n",
              "| Cenário | Runs | Rounds com configuração nociva | Quedas | Quedas em round nocivo "
              "| Rounds até voltar a 5 pp do valor anterior |",
              "|---|---:|---:|---:|---:|---|"]
    for scn in BROADCAST:
        r = broadcast(scn)
        if r:
            rec = ", ".join("—" if v is None else str(v) for v in r["recovery"]) or "—"
            lines.append(f"| `{scn}` | {r['runs']} | {r['harmful']}/{r['rounds']} | {r['drops']} "
                         f"| {r['drops_harmful']}/{r['drops']} | {rec} |")
    # the harmful set was read off the broadcast drops of seeds 0-4; seeds 5-9 came later
    lines += ["\n## Conjunto nocivo dentro e fora da amostra (seeds 0–4 vs 5–9)\n",
              "| Cenário | Seeds | Rounds com configuração nociva | Quedas | Quedas em round nocivo |",
              "|---|---|---:|---:|---:|"]
    for scn in ("ga_broadcast_cifar", "tpe_broadcast_cifar", "rs_broadcast_cifar"):
        for label, seeds in (("0–4", range(0, 5)), ("5–9", range(5, 10))):
            r = broadcast(scn, seeds)
            if r:
                lines.append(f"| `{scn}` | {label} | {r['harmful']}/{r['rounds']} | {r['drops']} "
                             f"| {r['drops_harmful']}/{r['drops']} |")
    lines += ["\n## FedEx: concentração da distribuição do servidor\n",
              "| Seed | Prob. da HP mais provável no round 10 | No round 20 | HP mais provável no round 20 |",
              "|---:|---:|---:|---|"]
    with SUMMARY.open(encoding="utf-8") as fh:
        fedex_runs = sorted((int(r["seed"]), r["run_id"]) for r in csv.DictReader(fh)
                            if r["scenario_name"] == "fedex_cifar" and r["status"] == "ok")
    for seed, rid in fedex_runs:
        with (RUNS_DIR / rid / "fedex_rounds.csv").open(encoding="utf-8") as fh:
            by_round = {int(r["server_round"]): r for r in csv.DictReader(fh)}
        last = by_round[max(by_round)]
        hp = f"{last['mle_optimizer']}, lr {last['mle_lr']}, batch {last['mle_batch_size']}"
        lines.append(f"| {seed} | {float(by_round[10]['fedex-mle-prob']):.2f} "
                     f"| {float(last['fedex-mle-prob']):.2f} | {hp} |")
    text = "\n".join(lines) + "\n"
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
        print(f"[agreement] wrote {args.markdown}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
