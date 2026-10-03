"""FedPop must evolve toward the better HPs and weights, credit the right client, and keep sampling paired."""
import random
from types import SimpleNamespace

import numpy as np

from federatedgeneticalgorithm import server_app
from federatedgeneticalgorithm.federated_fedpop import FedPop, incumbent

# one batch size and eps 0 make a "perturbed copy" an exact copy, so the tests can tell where it came from
GRID = {
    "batch_sizes": [64],
    "optimizers": ["sgd", "adam"],
    "learning_rates": [0.0005, 0.001, 0.003, 0.005, 0.01],
    "weight_decays": [0.0, 1e-4],
    "momentums": [0.9],
}
LRS = GRID["learning_rates"]


def _pop(num_configs=5):
    pop = FedPop(GRID, num_configs=num_configs, num_clients=5, seed=0, eps0=0.0, resample0=0.0)
    for proc in pop.processes:
        proc.locals = [dict(proc.base, lr=lr) for lr in LRS]
    return pop


def test_local_step_replaces_the_worst_client_hps_with_copies_of_the_best():
    pop = _pop()
    proc = pop.processes[0]
    pop.local_step(proc, [0.1, 0.9, 0.2, 0.8, 0.5], rnd=1, num_rounds=20)
    lrs = [hp["lr"] for hp in proc.locals]
    assert [lrs[0], lrs[2], lrs[4]] == [LRS[0], LRS[2], LRS[4]]
    assert lrs[1] in (LRS[0], LRS[2]) and lrs[3] in (LRS[0], LRS[2])


def test_global_step_gives_the_worst_processes_the_weights_of_the_best():
    pop = _pop()
    for i, (proc, loss) in enumerate(zip(pop.processes, [0.1, 0.9, 0.2, 0.8, 0.5])):
        proc.arrays, proc.history = f"w{i}", [loss]
    pop.global_step(rnd=2, num_rounds=20)
    weights = [p.arrays for p in pop.processes]
    assert [weights[0], weights[2], weights[4]] == ["w0", "w2", "w4"]
    assert weights[1] in ("w0", "w2") and weights[3] in ("w0", "w2")


def test_reported_process_has_the_lowest_validation_loss():
    assert incumbent([0.7, None, 0.3, 0.9]) == 2


def _reply(node_id, loss):
    return SimpleNamespace(metadata=SimpleNamespace(src_node_id=node_id),
                           content={"metrics": {"fedpop-val-loss": loss}}, has_error=lambda: False)


def test_losses_are_credited_to_the_slot_of_the_node_that_trained_it():
    node_ids = [11, 22, 33]
    losses, _ = server_app.fedpop_losses(node_ids, [_reply(33, 0.3), _reply(11, 0.1), _reply(22, 0.2)])
    assert losses == [0.1, 0.2, 0.3]


def test_population_leaves_the_global_rngs_alone():
    # client sampling uses the global `random`; touching it would unpair designs with the same seed
    random.seed(7)
    np.random.seed(7)
    py_state, np_state = random.getstate(), np.random.get_state()
    pop = FedPop(GRID, num_configs=5, num_clients=5, seed=3)
    for rnd in range(1, 5):
        for proc in pop.processes:
            pop.local_step(proc, list(np.linspace(0.1, 0.9, 5)), rnd, 20)
            proc.history.append(rnd / 10)
        pop.global_step(rnd, 20)
    assert random.getstate() == py_state
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), np_state))
