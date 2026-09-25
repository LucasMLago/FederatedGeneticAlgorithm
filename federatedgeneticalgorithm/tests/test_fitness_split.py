"""Server-side search must not read the test partitions."""
import pytest
import torch
from flwr.app import MetricRecord
from flwr.serverapp.strategy import FedAvg
from torch.utils.data import Subset, TensorDataset

from federatedgeneticalgorithm.config import config
from federatedgeneticalgorithm import server_app, task


class _RecordingSearcher:
    generation = 0
    current_idx = 0
    best_fitness = 0.0
    best_hp = {}
    population = []

    def __init__(self):
        self.recorded = []

    def record_fitness(self, fitness):
        self.recorded.append(fitness)
        return {"generation": 0}


class _Reply:
    def __init__(self, metrics):
        self.content = {"metrics": metrics}

    def has_error(self):
        return False


@pytest.fixture
def strategy(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_TELEMETRY_EXPORT", False)
    monkeypatch.setattr(config, "FED_FITNESS_SPLIT", "val")
    # eval-acc far from the val values so a mix-up shows
    monkeypatch.setattr(FedAvg, "aggregate_evaluate", lambda self, rnd, replies: MetricRecord({"eval-acc": 0.9}))
    s = server_app.FederatedGAFedAvg(_RecordingSearcher(), fraction_train=0.5)
    s._current_hp = {"lr": 0.01}
    return s


def test_broadcast_fitness_is_pooled_heldout_accuracy_not_test(strategy):
    replies = [
        _Reply({"eval-acc": 0.9, "fitness-val-acc": 0.2, "fitness-val-loss": 2.0, "fitness-val-num-examples": 100}),
        _Reply({"eval-acc": 0.9, "fitness-val-acc": 0.6, "fitness-val-loss": 1.0, "fitness-val-num-examples": 300}),
    ]
    strategy.aggregate_evaluate(1, replies)
    # (0.2*100 + 0.6*300) / 400; test would be 0.9, plain mean 0.4
    assert strategy.fed_ga.recorded == [pytest.approx(0.5)]


def test_missing_heldout_metric_refuses_instead_of_falling_back_to_test(strategy):
    replies = [_Reply({"eval-acc": 0.9})]
    with pytest.raises(RuntimeError, match="fitness-val-acc"):
        strategy.aggregate_evaluate(1, replies)
    assert strategy.fed_ga.recorded == []


@pytest.mark.parametrize("batch_size", [64, 128])
def test_heldout_split_is_exactly_what_local_training_leaves_out(batch_size):
    data = TensorDataset(torch.arange(500).float().unsqueeze(1), torch.zeros(500, dtype=torch.long))
    partition = Subset(data, list(range(7, 500, 3)))

    train_loader, val_loader, _ = task.build_dataloaders(partition, data, batch_size=batch_size, seed=3)
    trained_on = {partition.indices[i] for i in train_loader.dataset.indices}
    held_out_by_training = [partition.indices[i] for i in val_loader.dataset.indices]

    fitness_split = task.heldout_val_set(partition, seed=3).indices

    assert sorted(fitness_split) == sorted(held_out_by_training)
    assert trained_on.isdisjoint(fitness_split)
