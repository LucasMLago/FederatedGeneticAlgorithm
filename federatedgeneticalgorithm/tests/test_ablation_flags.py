"""Ablation switches: private surrogate pools and the broadcast GA without elitism."""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from federatedgeneticalgorithm import telemetry
from federatedgeneticalgorithm.config import config
from federatedgeneticalgorithm.federated_genetic_algorithm import FederatedGA
from federatedgeneticalgorithm.genetic_algorithm import HYPERPARAMS, GeneticAlgorithm, read_shared_pool


def test_private_pool_never_mixes_clients(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENABLE_SURROGATE_GA", True)
    monkeypatch.setattr(config, "SURROGATE_SHARED_POOL", False)
    monkeypatch.setattr(config, "ENABLE_TELEMETRY_EXPORT", False)
    monkeypatch.setattr(telemetry, "get_run_dir", lambda *a, **k: Path(tmp_path))
    monkeypatch.setattr(GeneticAlgorithm, "_evaluate_rung", lambda self, ind, gs, *a, **kw: (ind["lr"], 0.0, ind["lr"]))
    random.seed(0)
    np.random.seed(0)
    data = TensorDataset(torch.zeros(8, 1), torch.zeros(8, dtype=torch.long))
    for client in (0, 1):
        ga = GeneticAlgorithm(torch.nn.Linear(1, 1), data, data)
        for _ in range(4):
            ga.run_round_updates({}, client_id=client)
    for client in (0, 1):
        pool = read_shared_pool(str(tmp_path / f"hp_pool_client_{client}.pkl"))
        assert pool and {e["client_id"] for e in pool} == {client}
    assert not (tmp_path / "shared_hp_pool.pkl").exists()


def _ga_with_fixed_offspring(elitism):
    ga = FederatedGA(hyperparams=HYPERPARAMS, pop_size=4, seed=0, elitism=elitism)
    ga.best_hp = {"batch_size": 64, "optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0, "momentum": 0.9}
    ga.fitnesses = [0.5, 0.6, 0.7, 0.8]
    child = {"batch_size": 128, "optimizer": "adam", "lr": 0.001, "weight_decay": 0.0, "momentum": 0.5}
    ga._tournament = lambda: dict(child)
    ga._crossover = lambda a, b: dict(child)
    ga._mutate = lambda hp: dict(child)
    ga._evolve()
    return ga


def test_elitism_switch_controls_carrying_the_best():
    assert _ga_with_fixed_offspring(True).population[0] == _ga_with_fixed_offspring(True).best_hp
    assert all(hp["optimizer"] == "adam" for hp in _ga_with_fixed_offspring(False).population)
