#!/usr/bin/env python3
"""Score the broadcast GA's generation-0 candidates from a common checkpoint (cold and warm).

Plain PyTorch, no Flower/Ray; same partitions, train/test and FedAvg weighting as the simulation.
Usage: federatedgeneticalgorithm/.venv/bin/python scripts/expert_position_checkpoint.py --seeds 0 1 2
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from federatedgeneticalgorithm.config import config
from federatedgeneticalgorithm import task
from federatedgeneticalgorithm.genetic_algorithm import HYPERPARAMS
from federatedgeneticalgorithm.federated_genetic_algorithm import FederatedGA

REPO = Path(__file__).resolve().parents[1]
NUM_CLIENTS = 10
CLIENTS_PER_ROUND = 5  # FRACTION_TRAIN 0.5
EXPERT = {"batch_size": 64, "optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0, "momentum": 0.9}
FIELDS = [
    "seed", "phase", "checkpoint", "position", "is_expert", "batch_size", "optimizer", "lr",
    "weight_decay", "momentum", "clients", "ckpt_val_acc", "ckpt_test_acc", "val_acc", "test_acc", "seconds",
]


def _clone(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def fedavg(states, weights):
    total = float(sum(weights))
    out = {}
    for k, t0 in states[0].items():
        acc = sum(s[k].double() * (w / total) for s, w in zip(states, weights))
        out[k] = acc.to(t0.dtype) if t0.is_floating_point() else torch.round(acc).to(t0.dtype)
    return out


class Federation:
    def __init__(self, seed, device, smoke):
        self.seed = seed
        self.device = device
        self.train_parts, self.test_parts, self.val_parts = [], [], []
        for pid in range(NUM_CLIENTS):
            tr = task.get_partition(task.trainset, pid, NUM_CLIENTS, seed=seed)
            te = task.get_partition(task.testset, pid, NUM_CLIENTS, seed=seed, force_iid=True)
            if smoke:
                tr, te = Subset(tr.dataset, tr.indices[:160]), Subset(te.dataset, te.indices[:64])
            self.train_parts.append(tr)
            self.test_parts.append(te)
            self.val_parts.append(task.heldout_val_set(tr, seed=seed))

    def local_update(self, global_state, pid, hp):
        model = task.build_model()
        model.load_state_dict(global_state)
        trainloader, _, _ = task.build_dataloaders(
            self.train_parts[pid], self.test_parts[pid], batch_size=hp["batch_size"], seed=self.seed
        )
        task.train(
            model, trainloader, config.LOCAL_EPOCHS, hp["lr"], self.device, hp["optimizer"],
            hp["weight_decay"], hp.get("momentum", 0.0), mu=config.LOCAL_TRAIN_MU, global_state_dict=global_state,
        )
        return _clone(model.state_dict()), len(trainloader.dataset)

    def round(self, global_state, clients, hp):
        updates = [self.local_update(global_state, pid, hp) for pid in clients]
        return fedavg([u[0] for u in updates], [u[1] for u in updates])

    def evaluate(self, state):
        """(val acc, test acc), pooled over clients."""
        model = task.build_model()
        model.load_state_dict(state)
        out = []
        for parts in (self.val_parts, self.test_parts):
            correct = total = 0
            for ds in parts:
                _, acc = task.test(model, DataLoader(ds, batch_size=128, shuffle=False, num_workers=2), self.device)
                correct += acc * len(ds)
                total += len(ds)
            out.append(correct / total)
        return out[0], out[1]


def run_seed(seed, device, writer, smoke):
    config.SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cold = _clone(task.build_model().state_dict())
    fed = Federation(seed, device, smoke)
    ga = FederatedGA(
        hyperparams=HYPERPARAMS, pop_size=config.FED_GA_POPULATION_SIZE, mutation_prob=config.MUTATION_PROB,
        crossover_prob=config.CROSSOVER_PROB, tournament_size=config.TOURNAMENT_SIZE, seed=seed,
        seed_individuals=[dict(EXPERT)],
    )
    candidates = [dict(hp) for hp in ga.population]  # expert is first
    rng = np.random.default_rng(seed + 1000)

    def emit(phase, ckpt_name, pos, hp, clients, ckpt_scores, scores, t0):
        writer.writerow({
            "seed": seed, "phase": phase, "checkpoint": ckpt_name, "position": pos, "is_expert": int(pos == 1),
            **{k: hp[k] for k in ("batch_size", "optimizer", "lr", "weight_decay", "momentum")},
            "clients": " ".join(map(str, clients)), "ckpt_val_acc": round(ckpt_scores[0], 4),
            "ckpt_test_acc": round(ckpt_scores[1], 4), "val_acc": round(scores[0], 4),
            "test_acc": round(scores[1], 4), "seconds": round(time.perf_counter() - t0, 1),
        })
        print(f"[seed {seed}] {phase:<10} {ckpt_name:<5} pos={pos} {hp['optimizer']}/lr={hp['lr']} "
              f"val={scores[0]:.4f} test={scores[1]:.4f}", flush=True)

    # replay gen 0 in order (as in the real run); the result is the warm checkpoint
    state = cold
    state_scores = fed.evaluate(state)
    for pos, hp in enumerate(candidates, 1):
        t0 = time.perf_counter()
        clients = sorted(int(c) for c in rng.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False))
        new_state = fed.round(state, clients, hp)
        new_scores = fed.evaluate(new_state)
        emit("sequential", "-", pos, hp, clients, state_scores, new_scores, t0)
        state, state_scores = new_state, new_scores
    warm = state

    # every candidate from the same weights and clients
    clients = sorted(int(c) for c in rng.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False))
    for ckpt_name, ckpt in (("cold", cold), ("warm", warm)):
        ckpt_scores = fed.evaluate(ckpt)
        for pos, hp in enumerate(candidates, 1):
            t0 = time.perf_counter()
            emit("common", ckpt_name, pos, hp, clients, ckpt_scores, fed.evaluate(fed.round(ckpt, clients, hp)), t0)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--out", type=Path, default=REPO / "federatedgeneticalgorithm/artifacts/expert_position.csv")
    p.add_argument("--smoke", action="store_true", help="Tiny partitions and 1 local epoch, to check the code path.")
    args = p.parse_args()
    if args.smoke:
        config.LOCAL_EPOCHS = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    with args.out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for seed in args.seeds:
            run_seed(seed, device, writer, args.smoke)
            fh.flush()


if __name__ == "__main__":
    main()
