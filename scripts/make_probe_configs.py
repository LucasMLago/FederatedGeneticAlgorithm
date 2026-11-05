#!/usr/bin/env python3
"""Write the fixed-HP probe configs around the expert, to see whether anything beats it.

Each probe is `configs/fixed_expert_cifar.yaml` with one or two DEFAULT_* values changed. The first five
stay inside the GA's grid; the last three go outside it (lr 0.02, weight decay 5e-4).

Usage:
    uv run --project federatedgeneticalgorithm python scripts/make_probe_configs.py OUT_DIR
"""
import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
EXPERT_YAML = REPO / "configs" / "fixed_expert_cifar.yaml"

PROBES = {
    "probe_sgd_lr0p005": {"DEFAULT_LR": 0.005},
    "probe_sgd_bs128": {"DEFAULT_BATCH_SIZE": 128},
    "probe_sgd_wd1e-4": {"DEFAULT_WEIGHT_DECAY": 0.0001},
    "probe_sgd_wd1e-3": {"DEFAULT_WEIGHT_DECAY": 0.001},
    "probe_sgd_mom0p95": {"DEFAULT_MOMENTUM": 0.95},
    # outside the grid
    "probe_sgd_lr0p02": {"DEFAULT_LR": 0.02},
    "probe_sgd_wd5e-4": {"DEFAULT_WEIGHT_DECAY": 0.0005},
    "probe_sgd_lr0p02_wd5e-4": {"DEFAULT_LR": 0.02, "DEFAULT_WEIGHT_DECAY": 0.0005},
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(EXPERT_YAML.read_text(encoding="utf-8"))
    for name, changes in PROBES.items():
        cfg = {**base, "name": name, "overrides": {**base["overrides"], **changes}}
        cfg["description"] = "Expert HPs with " + ", ".join(f"{k}={v}" for k, v in changes.items()) + ". CIFAR-10, alpha=0.5."
        (args.out_dir / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"wrote {args.out_dir / (name + '.yaml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
