#!/usr/bin/env python3
"""Orchestrate a (config × seed) sweep through the runner CLI.

Runs are launched sequentially because the project's GPU budget is one T4.
For each (config, seed) pair we:
  1. Shell out to `federatedgeneticalgorithm.runner` with the requested seed.
  2. Find the resulting run_id from telemetry's active_run_id.txt at start
     time (or by directory mtime as a fallback).
  3. Read the server-aggregated CSV to extract peak / final eval accuracy.
  4. Append a row to the summary CSV so a partially completed sweep is still
     analyzable.

By default the orchestrator continues past individual run failures and
exits non-zero only if every run failed. The summary CSV doubles as a
resume marker: re-launching with the same path skips (config, seed) pairs
already present unless --force is given.

Usage:
    python scripts/run_matrix.py \\
        --configs configs/ga_broadcast_cifar.yaml \\
        --seeds 0 1 2 3 4 \\
        --federation local-simulation-gpu \\
        --tag headline-batch \\
        --summary artifacts/matrix_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "federatedgeneticalgorithm"
ARTIFACTS = APP_DIR / "artifacts"
RUNS_DIR = ARTIFACTS / "runs"

SUMMARY_HEADERS = [
    "config",
    "seed",
    "scenario_name",
    "run_id",
    "status",          # "ok" | "failed" | "skipped"
    "returncode",
    "peak_eval_acc",
    "final_eval_acc",
    "num_eval_rounds",
    "wall_seconds",
    "git_sha",
    "tag",
    "started_at",
    "finished_at",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", required=True, type=Path,
                   help="YAML configs to sweep.")
    p.add_argument("--seeds", nargs="+", required=True, type=int,
                   help="Seeds to use for each config.")
    p.add_argument("--federation", default="local-simulation-gpu",
                   help="Flower federation (default: local-simulation-gpu).")
    p.add_argument("--tag", default=None, help="Free-form label forwarded to runner --tag.")
    p.add_argument("--summary", type=Path, default=ARTIFACTS / "matrix_summary.csv",
                   help="Where to write/append the summary CSV (default: artifacts/matrix_summary.csv).")
    p.add_argument("--force", action="store_true",
                   help="Re-run (config, seed) pairs already present in the summary CSV.")
    p.add_argument("--min-eval-rounds", type=int, default=0,
                   help="Mark a run 'failed' unless it parsed at least this many eval rounds. "
                        "Use the config's NUM_SERVER_ROUNDS to reject runs shredded mid-way "
                        "(e.g. Ray actors OOM-killed on a shared host); failed pairs are "
                        "retried on the next relaunch.")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Halt the sweep on the first failure instead of continuing.")
    return p.parse_args()


def _scenario_name_from_yaml(path: Path) -> str:
    try:
        import yaml  # local import so the script can --help without pyyaml.
        with path.open() as fh:
            return (yaml.safe_load(fh) or {}).get("name", path.stem)
    except Exception:
        return path.stem


def _read_existing_summary(path: Path) -> set[tuple[str, int]]:
    """Return (config_str, seed) pairs that already completed successfully.

    Failed rows do NOT count as done: a relaunch of the same sweep retries
    them (they stay in the CSV as a historical record of the failure).
    """
    if not path.exists():
        return set()
    seen: set[tuple[str, int]] = set()
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("status") == "ok":
                try:
                    seen.add((row["config"], int(row["seed"])))
                except (KeyError, ValueError):
                    continue
    return seen


def _open_summary(path: Path):
    """Open the summary CSV in append mode, writing the header if new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    fh = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=SUMMARY_HEADERS)
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def _latest_run_id_after(reference_epoch: float) -> Optional[str]:
    """Return the run_id whose directory was created after `reference_epoch`.

    The runner kicks off a fresh run via initialize_run(force_new=True), so the
    most recent dir mtime > reference_epoch is the one we just launched.
    """
    if not RUNS_DIR.exists():
        return None
    candidates = []
    for entry in RUNS_DIR.iterdir():
        if entry.is_dir() and entry.stat().st_mtime >= reference_epoch:
            candidates.append((entry.stat().st_mtime, entry.name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _summarize_run(run_id: str) -> dict[str, object]:
    """Pull peak / final eval acc and git_sha / tag from the run directory."""
    run_dir = RUNS_DIR / run_id
    peak = final = None
    num_eval = 0
    server_csv = run_dir / "server_aggregated_rounds.csv"
    if server_csv.exists():
        with server_csv.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("phase") != "evaluate":
                    continue
                acc_raw = row.get("eval-acc") or row.get("val-accuracy")
                if not acc_raw:
                    continue
                try:
                    acc = float(acc_raw)
                except ValueError:
                    continue
                num_eval += 1
                final = acc
                peak = acc if peak is None else max(peak, acc)

    git_sha = tag = None
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            git_sha = meta.get("git_sha")
            tag = meta.get("tag")
        except json.JSONDecodeError:
            pass

    return {
        "peak_eval_acc": peak,
        "final_eval_acc": final,
        "num_eval_rounds": num_eval,
        "git_sha": git_sha,
        "tag": tag,
    }


def _run_one(config: Path, seed: int, federation: str, tag: Optional[str],
             min_eval_rounds: int = 0) -> dict[str, object]:
    started_wall = time.monotonic()
    started_iso = datetime.now().isoformat(timespec="seconds")
    reference_epoch = time.time()

    cmd = [
        sys.executable,
        "-m",
        "federatedgeneticalgorithm.runner",
        "--config",
        str(config.resolve()),
        "--seed",
        str(seed),
        "--federation",
        federation,
    ]
    if tag:
        cmd += ["--tag", tag]
    print(f"\n[matrix] launching: {config.name} seed={seed} federation={federation}", flush=True)
    print(f"[matrix] $ {' '.join(cmd)}", flush=True)

    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    wall = time.monotonic() - started_wall
    finished_iso = datetime.now().isoformat(timespec="seconds")
    run_id = _latest_run_id_after(reference_epoch) or ""
    metrics = _summarize_run(run_id) if run_id else {
        "peak_eval_acc": None,
        "final_eval_acc": None,
        "num_eval_rounds": 0,
        "git_sha": None,
        "tag": tag,
    }
    # "ok" requires more than returncode 0: `flwr run` exits 0 even when the
    # simulation backend fails to launch (e.g. flower-simulation not on PATH),
    # which would poison the resume marker with empty rows.
    num_eval = int(metrics.get("num_eval_rounds") or 0)
    launched_ok = (proc.returncode == 0 and bool(run_id)
                   and num_eval > 0 and num_eval >= min_eval_rounds)
    return {
        "config": str(config),
        "seed": seed,
        "scenario_name": _scenario_name_from_yaml(config),
        "run_id": run_id,
        "status": "ok" if launched_ok else "failed",
        "returncode": proc.returncode,
        "wall_seconds": round(wall, 1),
        "started_at": started_iso,
        "finished_at": finished_iso,
        **metrics,
    }


def main() -> int:
    args = parse_args()

    for cfg in args.configs:
        if not cfg.exists():
            print(f"[matrix] ERROR: config not found: {cfg}", file=sys.stderr)
            return 2

    pairs = list(product(args.configs, args.seeds))
    already = set() if args.force else _read_existing_summary(args.summary)
    pending = [(c, s) for (c, s) in pairs if (str(c), s) not in already]

    print(f"[matrix] summary file: {args.summary}", flush=True)
    print(f"[matrix] {len(pending)}/{len(pairs)} pair(s) pending "
          f"({len(pairs) - len(pending)} skipped, already in summary)", flush=True)
    if not pending:
        return 0

    fh, writer = _open_summary(args.summary)
    any_failed = False
    all_failed = True
    try:
        for i, (config, seed) in enumerate(pending, start=1):
            print(f"\n=== [{i}/{len(pending)}] {config.name} seed={seed} ===", flush=True)
            row = _run_one(config, seed, args.federation, args.tag,
                           min_eval_rounds=args.min_eval_rounds)
            writer.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())

            if row["status"] == "ok":
                all_failed = False
                print(f"[matrix] OK peak={row['peak_eval_acc']} final={row['final_eval_acc']} "
                      f"wall={row['wall_seconds']}s run_id={row['run_id']}", flush=True)
            else:
                any_failed = True
                print(f"[matrix] FAILED rc={row['returncode']} run_id={row['run_id']}", flush=True)
                if args.stop_on_error:
                    print("[matrix] --stop-on-error set; halting sweep.", flush=True)
                    break
    finally:
        fh.close()

    if all_failed and any_failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
