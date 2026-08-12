from __future__ import annotations

import csv
import json
import fcntl
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from federatedgeneticalgorithm.config import config


ACTIVE_RUN_FILE = "active_run_id.txt"
RUNS_FOLDER = "runs"


def _pid_alive(pid: int) -> bool:
    """Return True iff the given pid points at a live process owned by this uid.

    `os.kill(pid, 0)` does not deliver a signal; it only validates that the pid
    exists and we have permission to signal it.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user; treat as alive so we
        # don't stomp someone else's run dir.
        return True
    return True


def _parse_active_run(raw: str) -> Tuple[Optional[int], str]:
    """Parse `<pid>:<run_id>` (current format) or `<run_id>` (legacy)."""
    stripped = raw.strip()
    if ":" in stripped:
        pid_part, run_part = stripped.split(":", 1)
        try:
            return int(pid_part), run_part.strip()
        except ValueError:
            return None, stripped
    return None, stripped


def _git_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _archive_runner_inputs(run_dir: Path) -> None:
    """Copy the YAML config + resolved JSON the runner CLI passed via env.

    Both files persist in the run dir so the experiment is reproducible from
    the archived state alone, without needing the runner's tempfile to survive.
    """
    yaml_src = os.environ.get("FGA_CONFIG_YAML_PATH")
    if yaml_src and Path(yaml_src).exists():
        shutil.copy(yaml_src, run_dir / "config.yaml")

    json_src = os.environ.get("FGA_CONFIG_PATH")
    if json_src and Path(json_src).exists():
        shutil.copy(json_src, run_dir / "resolved_config.json")

SERVER_AGGREGATED_HEADERS = [
    "timestamp",
    "run_id",
    "server_round",
    "phase",
    "num_replies",
    "train-loss",
    "train-accuracy",
    "val-loss",
    "val-accuracy",
    "hp_lr",
    "hp_batch_size",
    "hp_weight_decays",
    "hp_momentum",
    "eval-loss",
    "eval-acc",
]

CLIENT_ROUND_HEADERS = [
    "timestamp",
    "run_id",
    "client_id",
    "client_round",
    "train-loss",
    "train-accuracy",
    "val-loss",
    "val-accuracy",
    "local-test-loss",
    "local-test-accuracy",
    "ga_best_fitness",
    "batch_size",
    "optimizer",
    "lr",
    "weight_decay",
    "momentum",
    "num-examples",
    "ga_ran",
    "ga_time_s",
    "local_train_time_s",
    "total_visit_time_s",
]

GA_CANDIDATES_HEADERS = [
    "timestamp",
    "run_id",
    "client_id",
    "client_round",
    "candidate_index",
    "rung",
    "fitness",
    "batch_size",
    "optimizer",
    "lr",
    "weight_decay",
    "momentum",
]

RUN_SCENARIO_HEADERS = [
    "run_id",
    "scenario",
    "ga_enabled",
    "surrogate_enabled",
    "created_at",
]


def _base_dir() -> Path:
    """Resolve the telemetry base directory to an absolute path.

    Ray actor workers run with cwd under `/tmp/ray/...`, so a relative
    `Path("artifacts")` silently lands their CSV writes in the wrong tree.
    Resolution order:
      1. `FGA_TELEMETRY_BASE` env var (set by the runner CLI for reproducibility)
      2. `config.TELEMETRY_BASE_DIR` if it is already absolute
      3. `<flwr-app-dir>/<TELEMETRY_BASE_DIR>` anchored off this module's path
         (parents[1] is the Flower app dir containing pyproject.toml)
    """
    env_base = os.environ.get("FGA_TELEMETRY_BASE")
    if env_base:
        return Path(env_base).expanduser().resolve()

    configured = Path(config.TELEMETRY_BASE_DIR)
    if configured.is_absolute():
        return configured

    app_root = Path(__file__).resolve().parents[1]
    return app_root / configured


def _active_run_path() -> Path:
    return _base_dir() / ACTIVE_RUN_FILE


def _runs_root() -> Path:
    return _base_dir() / RUNS_FOLDER


def _scenario_name() -> str:
    return f"ga_{int(bool(config.ENABLE_GA))}_surrogate_{int(bool(config.ENABLE_SURROGATE_GA))}"


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _append_csv_row(csv_path: Path, headers: list[str], row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a+", newline="", encoding="utf-8") as file_obj:
        if fcntl is not None:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)

        file_obj.seek(0, 2)
        should_write_header = file_obj.tell() == 0

        writer = csv.DictWriter(file_obj, fieldnames=headers)
        if should_write_header:
            writer.writeheader()

        prepared_row = {header: _coerce_scalar(row.get(header)) for header in headers}
        writer.writerow(prepared_row)

        if fcntl is not None:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


def initialize_run(force_new: bool = False, record_config: bool = False) -> str:
    """Create or reuse a run_id shared by server and clients in the same execution.

    The active-run file stores `<pid>:<run_id>` so we can detect zombie state:
    if the recorded writer pid is no longer alive, treat the existing entry as
    stale and start a fresh run instead of silently appending to the wrong dir.
    Ray actor workers (different pid) will still reuse the run_id because the
    writer pid (the server process) is alive.
    """
    _base_dir().mkdir(parents=True, exist_ok=True)
    _runs_root().mkdir(parents=True, exist_ok=True)
    active_file = _active_run_path()

    existing_pid: Optional[int] = None
    existing_run_id: Optional[str] = None
    if active_file.exists():
        existing_pid, existing_run_id = _parse_active_run(
            active_file.read_text(encoding="utf-8")
        )

    must_create = (
        force_new
        or existing_run_id is None
        or (existing_pid is not None and not _pid_alive(existing_pid))
    )

    if must_create:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        active_file.write_text(f"{os.getpid()}:{run_id}", encoding="utf-8")
    else:
        run_id = existing_run_id  # type: ignore[assignment]

    run_dir = _runs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if record_config:
        meta_path = run_dir / "run_metadata.json"
        config_snapshot = {
            name: _coerce_scalar(getattr(config, name))
            for name in dir(config)
            if name.isupper() and not name.startswith("_")
        }
        metadata = {
            "run_id": run_id,
            "created_at": _timestamp(),
            "scenario": _scenario_name(),
            "ga_enabled": bool(config.ENABLE_GA),
            "surrogate_enabled": bool(config.ENABLE_SURROGATE_GA),
            "config": config_snapshot,
            "git_sha": _git_sha(),
            "tag": os.environ.get("FGA_RUN_TAG"),
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _archive_runner_inputs(run_dir)

    scenario_csv_path = run_dir / "run_scenario.csv"
    if not scenario_csv_path.exists():
        scenario_row = {
            "run_id": run_id,
            "scenario": _scenario_name(),
            "ga_enabled": bool(config.ENABLE_GA),
            "surrogate_enabled": bool(config.ENABLE_SURROGATE_GA),
            "created_at": _timestamp(),
        }
        _append_csv_row(scenario_csv_path, RUN_SCENARIO_HEADERS, scenario_row)

    return run_id


def get_run_id() -> str:
    return initialize_run(force_new=False, record_config=False)


def get_run_dir(run_id: Optional[str] = None) -> Path:
    current_run_id = run_id or get_run_id()
    run_dir = _runs_root() / current_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_partition_distribution(
    train_histograms: list, test_histograms: list, mode: str, alpha: float
) -> None:
    """Persist per-partition class distribution for the paper's figures."""
    run_dir = get_run_dir()
    payload = {
        "partition_mode": mode,
        "dirichlet_alpha": alpha,
        "train": [{str(k): v for k, v in hist.items()} for hist in train_histograms],
        "test": [{str(k): v for k, v in hist.items()} for hist in test_histograms],
    }
    (run_dir / "partition_distribution.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def append_server_aggregated_row(server_round: int, phase: str, num_replies: int, metrics: Dict[str, Any]) -> None:
    if not config.ENABLE_TELEMETRY_EXPORT:
        return

    run_id = get_run_id()
    run_dir = get_run_dir(run_id)
    csv_path = run_dir / "server_aggregated_rounds.csv"

    row = {
        "timestamp": _timestamp(),
        "run_id": run_id,
        "server_round": server_round,
        "phase": phase,
        "num_replies": num_replies,
    }
    row.update(metrics)

    _append_csv_row(csv_path, SERVER_AGGREGATED_HEADERS, row)


def append_client_round_row(client_id: int, client_round: int, metrics: Dict[str, Any], best_hp: Dict[str, Any], best_fitness: float) -> None:
    if not config.ENABLE_TELEMETRY_EXPORT:
        return

    run_id = get_run_id()
    run_dir = get_run_dir(run_id)
    csv_path = run_dir / "client_round_metrics.csv"

    row = {
        "timestamp": _timestamp(),
        "run_id": run_id,
        "client_id": client_id,
        "client_round": client_round,
        "ga_best_fitness": best_fitness,
        "batch_size": best_hp.get("batch_size"),
        "optimizer": best_hp.get("optimizer"),
        "lr": best_hp.get("lr"),
        "weight_decay": best_hp.get("weight_decay"),
        "momentum": best_hp.get("momentum", 0.0),
    }
    row.update(metrics)

    _append_csv_row(csv_path, CLIENT_ROUND_HEADERS, row)


def append_ga_candidates_rows(client_id: int, client_round: int, ga_entries: Iterable[Dict[str, Any]]) -> None:
    if not config.ENABLE_TELEMETRY_EXPORT or not config.ENABLE_GA_CANDIDATE_EXPORT:
        return

    run_id = get_run_id()
    run_dir = get_run_dir(run_id)
    csv_path = run_dir / "ga_candidates.csv"

    for candidate_index, entry in enumerate(ga_entries):
        hp = entry.get("hp", {})
        row = {
            "timestamp": _timestamp(),
            "run_id": run_id,
            "client_id": client_id,
            "client_round": client_round,
            "candidate_index": candidate_index,
            "rung": entry.get("rung"),
            "fitness": entry.get("fitness"),
            "batch_size": hp.get("batch_size"),
            "optimizer": hp.get("optimizer"),
            "lr": hp.get("lr"),
            "weight_decay": hp.get("weight_decay"),
            "momentum": hp.get("momentum", 0.0),
        }
        _append_csv_row(csv_path, GA_CANDIDATES_HEADERS, row)
