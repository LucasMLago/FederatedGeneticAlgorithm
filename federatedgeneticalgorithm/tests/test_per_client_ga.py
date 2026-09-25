"""Per-client GA (plain and surrogate): offspring must reach the population, scores must survive reloads."""
import random

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from federatedgeneticalgorithm.config import config
from federatedgeneticalgorithm.genetic_algorithm import GeneticAlgorithm


def _score(hp):
    # fake fitness: higher lr / batch 128 wins
    return hp["lr"] * 10 + (0.01 if hp["batch_size"] == 128 else 0.0)


def _sig(hp):
    return GeneticAlgorithm._hp_signature(hp)


@pytest.fixture
def new_ga(monkeypatch):
    monkeypatch.setattr(config, "GA_EVOLVE_POPULATION", True)
    monkeypatch.setattr(config, "ENABLE_SURROGATE_GA", False)
    monkeypatch.setattr(config, "ENABLE_TELEMETRY_EXPORT", False)
    monkeypatch.setattr(GeneticAlgorithm, "_evaluate_rung", lambda self, ind, gs, *a, **kw: (_score(ind), 0.0, _score(ind)))
    random.seed(0)
    np.random.seed(0)
    data = TensorDataset(torch.zeros(8, 1), torch.zeros(8, dtype=torch.long))
    return lambda: GeneticAlgorithm(torch.nn.Linear(1, 1), data, data)


def test_population_takes_in_offspring_and_never_loses_its_best(new_ga, tmp_path):
    state = tmp_path / "ga.pkl"
    ga = new_ga()
    ever_in_population = set()
    best_so_far = []
    for _ in range(12):
        ga.run_round_updates({}, client_id=0)
        ever_in_population |= {_sig(ind) for ind in ga.population}
        best_so_far.append(max(_score(ind) for ind in ga.population))
        ga.save_state(str(state))
        ga = new_ga()
        ga.load_state(str(state))
    # frozen population would stay at POPULATION_SIZE distinct HPs
    assert len(ever_in_population) > config.POPULATION_SIZE
    assert best_so_far == sorted(best_so_far)


def test_population_scores_survive_the_state_reload(new_ga, tmp_path):
    state = tmp_path / "ga.pkl"
    ga = new_ga()
    ga.run_round_updates({}, client_id=0)
    ga.save_state(str(state))
    reloaded = new_ga()
    reloaded.load_state(str(state))
    assert [ind.fitness.values[0] if ind.fitness.valid else None for ind in reloaded.population] == [
        _score(ind) for ind in ga.population
    ]


def test_surrogate_arm_population_evolves_too(new_ga, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_SURROGATE_GA", True)
    monkeypatch.setattr(GeneticAlgorithm, "_shared_pool_path", staticmethod(lambda: str(tmp_path / "pool.pkl")))
    state = tmp_path / "ga.pkl"
    ga = new_ga()
    ever_in_population = set()
    for _ in range(12):
        ga.run_round_updates({}, client_id=0)
        ever_in_population |= {_sig(ind) for ind in ga.population}
        ga.save_state(str(state))
        ga = new_ga()
        ga.load_state(str(state))
    assert len(ever_in_population) > config.POPULATION_SIZE
