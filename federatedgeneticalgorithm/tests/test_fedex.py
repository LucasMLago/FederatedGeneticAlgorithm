"""FedEx must learn from the right client and must not disturb the paired client sampling."""
import random
from types import SimpleNamespace

import numpy as np
import pytest

from federatedgeneticalgorithm.config import config
from federatedgeneticalgorithm import server_app
from federatedgeneticalgorithm.federated_baselines import FedEx

GRID = {
    "batch_sizes": [64],
    "optimizers": ["sgd", "adam"],
    "learning_rates": [0.001, 0.01],
    "weight_decays": [0.0],
    "momentums": [0.9],
}


def _lr_prob(fedex, lr):
    k = fedex._keys.index("learning_rates")
    return fedex.theta[k][GRID["learning_rates"].index(lr)]


def test_distribution_moves_toward_the_lower_validation_error():
    fedex = FedEx(GRID, seed=0, baseline_discount=0.5)
    for _ in range(5):
        assigned, errors = [], []
        for _ in range(6):
            idx, hp = fedex.sample()
            assigned.append(idx)
            errors.append(0.2 if hp["lr"] == 0.001 else 0.8)
        fedex.step(assigned, errors, [100] * len(assigned))
    assert _lr_prob(fedex, 0.001) > 0.9


def test_sampling_and_updates_leave_the_global_rngs_alone():
    # client sampling uses the global `random`; touching it would unpair designs with the same seed
    random.seed(7)
    np.random.seed(7)
    py_state, np_state = random.getstate(), np.random.get_state()
    fedex = FedEx(GRID, seed=3)
    for _ in range(4):
        picks = [fedex.sample() for _ in range(5)]
        fedex.step([idx for idx, _ in picks], [0.5, 0.4, 0.3, 0.2, 0.1], [1] * 5)
    assert random.getstate() == py_state
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), np_state))


class _Grid:
    def get_node_ids(self):
        return [11, 22]


def _reply(node_id, val_acc):
    return SimpleNamespace(
        metadata=SimpleNamespace(src_node_id=node_id),
        content={"metrics": {"fedex-val-acc": val_acc, "fedex-val-num-examples": 200}},
        has_error=lambda: False,
    )


def test_update_credits_each_error_to_the_node_that_trained_that_configuration(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_TELEMETRY_EXPORT", False)
    fedex = FedEx(GRID, seed=0, baseline_discount=0.0)
    def idx(lr):
        return tuple(GRID[k].index(lr) if k == "learning_rates" else 0 for k in fedex._keys)

    slow, fast = idx(0.001), idx(0.01)
    queue = [(slow, fedex._hp(slow)), (fast, fedex._hp(fast))]
    monkeypatch.setattr(fedex, "sample", lambda: queue.pop(0))

    strategy = server_app.FedExFedAvg(fedex, fraction_train=1.0, min_available_nodes=2, min_train_nodes=2)
    random.seed(0)
    messages = strategy.configure_train(1, server_app.ArrayRecord(), server_app.ConfigRecord({}), _Grid())
    sent = {m.metadata.dst_node_id: m.content["config"]["fed_ga_hp_lr"] for m in messages}
    slow_node = next(n for n, lr in sent.items() if lr == 0.001)
    fast_node = next(n for n, lr in sent.items() if lr == 0.01)
    assert set(sent.values()) == {0.001, 0.01}

    # replies come back in the opposite order from the messages
    replies = [_reply(fast_node, 0.9), _reply(slow_node, 0.3)] if messages[0].metadata.dst_node_id == slow_node \
        else [_reply(slow_node, 0.3), _reply(fast_node, 0.9)]
    strategy._after_aggregate_train(1, replies)

    assert _lr_prob(fedex, 0.01) > _lr_prob(fedex, 0.001)
