"""Pruned search space: with exclusion on no searcher trains a harmful HP; with it off they are unchanged."""
import hashlib
import json

import pytest

from federatedgeneticalgorithm.federated_baselines import FederatedRandomSearch, FederatedTPE
from federatedgeneticalgorithm.federated_genetic_algorithm import FederatedGA
from federatedgeneticalgorithm.genetic_algorithm import HYPERPARAMS
from federatedgeneticalgorithm.search_space import is_harmful

EXPERT = {"batch_size": 64, "optimizer": "sgd", "lr": 0.01, "weight_decay": 0.0, "momentum": 0.9}
SEARCHERS = {
    "ga": (FederatedGA, {"seed_individuals": [EXPERT]}),
    "rs": (FederatedRandomSearch, {"seed_individuals": [EXPERT]}),
    "tpe": (FederatedTPE, {}),
}
# HP sequences of the three searchers before the exclusion existed (main, 3a95d20); the paper's
# runs must stay reproducible with exclusion off
UNPRUNED_DIGESTS = {"ga": "780020936ad07742", "rs": "8959c9992208667e", "tpe": "c07265bd5c7071d8"}


def fitness(hp):
    # rewards high lr, which pulls GA and TPE toward the harmful region
    return hp["lr"] * 50 + (0.01 if hp["optimizer"] in ("lion", "adam") else 0.0)


def trace(name, rounds=20, seed=0, **extra):
    cls, kw = SEARCHERS[name]
    searcher = cls(hyperparams=HYPERPARAMS, pop_size=4, seed=seed, **kw, **extra)
    seq = []
    for r in range(1, rounds + 1):
        hp = searcher.select_for_round(r)
        seq.append(hp)
        searcher.record_fitness(fitness(hp))
    return seq, searcher


@pytest.mark.parametrize("name", sorted(SEARCHERS))
def test_excluded_searcher_never_selects_a_harmful_hp(name):
    for seed in range(3):
        unpruned, _ = trace(name, rounds=40, seed=seed)
        assert any(is_harmful(hp) for hp in unpruned), "fixture must reach the harmful region"
        pruned, _ = trace(name, rounds=40, seed=seed, exclude=is_harmful)
        assert not any(is_harmful(hp) for hp in pruned)


@pytest.mark.parametrize("name", sorted(SEARCHERS))
def test_exclusion_off_keeps_the_unpruned_sequence(name):
    seq, _ = trace(name)
    digest = hashlib.sha256(json.dumps(seq, sort_keys=True).encode()).hexdigest()[:16]
    assert digest == UNPRUNED_DIGESTS[name]
