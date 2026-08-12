"""Experiment runner CLI.

Loads a YAML scenario file, validates its overrides against the canonical
config module, writes a resolved JSON snapshot to a tempfile, exports the
env vars the rest of the codebase reads (`FGA_CONFIG_PATH`, `FGA_DATA_ROOT`,
`FGA_TELEMETRY_BASE`, `FGA_CONFIG_YAML_PATH`, `FGA_RUN_TAG`), and shells out
to `flwr run` so the experiment is reproducible from the YAML alone.

Usage:
    python -m federatedgeneticalgorithm.runner --config configs/exp_e.yaml --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

# Layout: <repo>/federatedgeneticalgorithm/federatedgeneticalgorithm/runner.py
PACKAGE_DIR = Path(__file__).resolve().parent           # .../federatedgeneticalgorithm/federatedgeneticalgorithm
APP_DIR = PACKAGE_DIR.parent                            # .../federatedgeneticalgorithm  (flwr app dir, has pyproject.toml)
REPO_ROOT = APP_DIR.parent                              # .../FederatedGeneticAlgorithm


def _known_config_keys() -> set[str]:
    """Names of UPPER_CASE constants defined in the canonical config module."""
    from federatedgeneticalgorithm.config import config as cfg_module

    return {
        name
        for name in dir(cfg_module)
        if name.isupper() and not name.startswith("_")
    }


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must parse to a mapping at the top level, got {type(data).__name__}")
    return data


def _validate_and_extract(cfg: Dict[str, Any], src_path: Path) -> Dict[str, Any]:
    """Verify schema and return the flat dict of overrides ready for JSON dump."""
    if "name" not in cfg or not isinstance(cfg["name"], str):
        raise ValueError(f"{src_path}: missing string 'name' field")
    if "overrides" not in cfg or not isinstance(cfg["overrides"], dict):
        raise ValueError(f"{src_path}: missing 'overrides' mapping")

    overrides = cfg["overrides"]
    unknown = sorted(set(overrides) - _known_config_keys())
    if unknown:
        raise ValueError(
            f"{src_path}: keys not defined in config.py: {unknown}. "
            "Add them to config.py first, or remove them from the YAML."
        )
    return dict(overrides)


def _build_env(
    json_snapshot_path: Path,
    yaml_config_path: Path,
    tag: str | None,
) -> Dict[str, str]:
    env = os.environ.copy()
    env["FGA_CONFIG_PATH"] = str(json_snapshot_path)
    env["FGA_CONFIG_YAML_PATH"] = str(yaml_config_path.resolve())
    env["FGA_DATA_ROOT"] = str((REPO_ROOT / "data").resolve())
    env["FGA_TELEMETRY_BASE"] = str((APP_DIR / "artifacts").resolve())
    if tag:
        env["FGA_RUN_TAG"] = tag
    return env


def _resolve_flwr_binary() -> Path:
    """Find the `flwr` console script in the same venv as the running python."""
    candidate = Path(sys.executable).parent / "flwr"
    if candidate.exists():
        return candidate
    raise RuntimeError(
        f"flwr CLI not found next to {sys.executable}. "
        "Run inside the project venv (e.g. `uv run python -m federatedgeneticalgorithm.runner ...`)."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m federatedgeneticalgorithm.runner",
        description="Run a FederatedGA experiment from a YAML config file.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML scenario file.")
    parser.add_argument("--seed", type=int, default=None, help="Override the SEED in the config.")
    parser.add_argument(
        "--federation",
        default="local-simulation-gpu",
        help="Flower federation name from pyproject.toml (default: local-simulation-gpu).",
    )
    parser.add_argument("--tag", default=None, help="Free-form label stored in run_metadata.json.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config and print the JSON that would be exported, without launching flwr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import subprocess

    args = parse_args(argv)

    cfg = _load_yaml(args.config)
    overrides = _validate_and_extract(cfg, args.config)
    if args.seed is not None:
        overrides["SEED"] = args.seed

    snapshot = tempfile.NamedTemporaryFile(
        prefix="fga_config_", suffix=".json", mode="w", delete=False, encoding="utf-8"
    )
    json.dump(overrides, snapshot, indent=2, sort_keys=True)
    snapshot.close()
    snapshot_path = Path(snapshot.name)

    if args.dry_run:
        print(f"# scenario: {cfg['name']}")
        print(f"# would write FGA_CONFIG_PATH={snapshot_path}")
        print(json.dumps(overrides, indent=2, sort_keys=True))
        snapshot_path.unlink(missing_ok=True)
        return 0

    env = _build_env(snapshot_path, args.config, args.tag)
    flwr = _resolve_flwr_binary()
    cmd = [str(flwr), "run", ".", args.federation]
    print(f"[runner] scenario={cfg['name']} seed={overrides.get('SEED')}", flush=True)
    print(f"[runner] $ {' '.join(cmd)}  (cwd={APP_DIR})", flush=True)

    rc = 1
    try:
        rc = subprocess.run(cmd, env=env, cwd=APP_DIR).returncode
    finally:
        # Keep the snapshot on failure so the user can inspect/replay it.
        if rc == 0:
            snapshot_path.unlink(missing_ok=True)
        else:
            print(f"[runner] flwr exited non-zero; snapshot kept at {snapshot_path}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
