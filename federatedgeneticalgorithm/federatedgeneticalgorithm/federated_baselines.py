"""Random Search and TPE with the FederatedGA interface, so FederatedGAFedAvg can run them as is.
FedEx is also here, but it assigns one HP per client, so it runs under FedExFedAvg instead."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from federatedgeneticalgorithm.federated_genetic_algorithm import KEY_MAP


@dataclass
class FederatedRandomSearch:
    """Uniform random HP each round; population and generation only exist to look like FederatedGA."""

    hyperparams: Dict[str, List]
    pop_size: int = 4  # only drives the generation bookkeeping
    seed: int = 0
    seed_individuals: Optional[List[Dict]] = None  # optionally plant known HPs in the initial pool

    history: List[Tuple[int, Dict, float]] = field(default_factory=list)
    population: List[Dict] = field(default_factory=list)
    fitnesses: List[Optional[float]] = field(default_factory=list)
    current_idx: int = 0
    generation: int = 0
    best_hp: Optional[Dict] = None
    best_fitness: float = float("-inf")
    _current_hp: Optional[Dict] = None
    _rng: random.Random = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._seed_queue: List[Dict] = list(self.seed_individuals) if self.seed_individuals else []
        # server_app prints the population at startup, so it can't be empty
        self.population = [self._sample_or_seed() for _ in range(self.pop_size)]
        self.fitnesses = [None] * self.pop_size

    def _sample_or_seed(self) -> Dict:
        if self._seed_queue:
            return dict(self._seed_queue.pop(0))
        return {
            KEY_MAP[k_plural]: self._rng.choice(values)
            for k_plural, values in self.hyperparams.items()
        }

    def select_for_round(self, server_round: int) -> Dict:
        self._current_hp = self._sample_or_seed()
        return dict(self._current_hp)

    def record_fitness(self, fitness: float) -> Dict[str, object]:
        hp = dict(self._current_hp) if self._current_hp is not None else {}
        self.history.append((self.generation, hp, fitness))
        is_new_best = fitness > self.best_fitness
        if is_new_best:
            self.best_fitness = fitness
            self.best_hp = hp
        # fake GA generations so the telemetry looks the same
        idx = self.current_idx
        self.population[idx] = hp
        self.fitnesses[idx] = fitness
        info = {
            "generation": self.generation,
            "individual": idx,
            "hp": hp,
            "fitness": fitness,
            "is_new_best": is_new_best,
        }
        self.current_idx = (self.current_idx + 1) % self.pop_size
        if self.current_idx == 0:
            self.generation += 1
            info["evolved_to_generation"] = self.generation
            self.fitnesses = [None] * self.pop_size
        return info


@dataclass
class FederatedTPE:
    """Optuna TPE over the same categorical grid as the GA."""

    hyperparams: Dict[str, List]
    pop_size: int = 4  # only drives the generation bookkeeping
    seed: int = 0
    seed_individuals: Optional[List[Dict]] = None  # accepted but ignored

    history: List[Tuple[int, Dict, float]] = field(default_factory=list)
    population: List[Dict] = field(default_factory=list)
    fitnesses: List[Optional[float]] = field(default_factory=list)
    current_idx: int = 0
    generation: int = 0
    best_hp: Optional[Dict] = None
    best_fitness: float = float("-inf")

    _study: object = field(default=None, repr=False)
    _trial: object = field(default=None, repr=False)
    _distributions: object = field(default=None, repr=False)
    _current_hp: Optional[Dict] = None

    def __post_init__(self) -> None:
        import optuna
        import optuna.distributions as D

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # default n_startup_trials=10, so half of a 20-round budget is random
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._distributions = {
            KEY_MAP[k_plural]: D.CategoricalDistribution(list(values))
            for k_plural, values in self.hyperparams.items()
        }
        # telemetry snapshot; separate RNG so no TPE trial is spent before round 1
        rng = random.Random(self.seed)
        self.population = [
            {KEY_MAP[k]: rng.choice(v) for k, v in self.hyperparams.items()}
            for _ in range(self.pop_size)
        ]
        self.fitnesses = [None] * self.pop_size
        if self.seed_individuals:
            # not supported, enqueue_trial doesn't mix well with TPE's own startup trials
            pass

    def select_for_round(self, server_round: int) -> Dict:
        self._trial = self._study.ask(self._distributions)
        self._current_hp = {name: self._trial.params[name] for name in self._distributions}
        return dict(self._current_hp)

    def record_fitness(self, fitness: float) -> Dict[str, object]:
        self._study.tell(self._trial, float(fitness))
        hp = dict(self._current_hp) if self._current_hp is not None else {}
        self.history.append((self.generation, hp, fitness))
        is_new_best = fitness > self.best_fitness
        if is_new_best:
            self.best_fitness = fitness
            self.best_hp = hp
        idx = self.current_idx
        self.population[idx] = hp
        self.fitnesses[idx] = fitness
        info = {
            "generation": self.generation,
            "individual": idx,
            "hp": hp,
            "fitness": fitness,
            "is_new_best": is_new_best,
        }
        self.current_idx = (self.current_idx + 1) % self.pop_size
        if self.current_idx == 0:
            self.generation += 1
            info["evolved_to_generation"] = self.generation
            self.fitnesses = [None] * self.pop_size
        return info


def _discounted_mean(trace: List[float], factor: float) -> float:
    weight = factor ** np.flip(np.arange(len(trace)), axis=0)
    return float(np.inner(trace, weight) / weight.sum())


class FedEx:
    """FedEx (Khodak et al., NeurIPS 2021) over a product of categoricals, one per HP.

    Follows the authors' reference code (github.com/mkhodak/FedEx, hyper.py) with its CIFAR
    defaults: eta0 = sqrt(2 log k) per HP, 'aggressive' step size, absolute validation error as
    the objective (diff=False), and a baseline discount drawn from U[0, 1). The RS/SHA wrapper that
    tunes server settings over many trajectories is left out: one run is one trajectory, as for
    the other searchers. Uses its own RNG, so client sampling (global `random`) stays paired.
    """

    def __init__(
        self,
        hyperparams: Dict[str, List],
        seed: int = 0,
        sched: str = "aggressive",
        baseline_discount: Optional[float] = None,
    ) -> None:
        if sched not in ("aggressive", "adaptive", "auto", "constant"):
            raise ValueError(f"unknown FedEx step-size schedule {sched!r}")
        self.hyperparams = hyperparams
        self._rng = np.random.default_rng(seed)
        self._keys = sorted(hyperparams)
        sizes = [len(hyperparams[k]) for k in self._keys]
        self._eta0 = [np.sqrt(2.0 * np.log(size)) for size in sizes]
        self._sched = sched
        self.baseline_discount = (
            float(self._rng.uniform(0.0, 1.0)) if baseline_discount is None else float(baseline_discount)
        )
        self._z = [np.full(size, -np.log(size)) for size in sizes]
        self.theta = [np.exp(z) for z in self._z]
        self._store = [0.0 for _ in sizes]
        self._refine_trace: List[float] = []

    def _hp(self, idx: Tuple[int, ...]) -> Dict:
        return {KEY_MAP[k]: self.hyperparams[k][i] for k, i in zip(self._keys, idx)}

    def sample(self) -> Tuple[Tuple[int, ...], Dict]:
        idx = tuple(int(self._rng.choice(len(t), p=t)) for t in self.theta)
        return idx, self._hp(idx)

    def mle(self) -> Dict:
        return self._hp(tuple(int(t.argmax()) for t in self.theta))

    def entropy(self) -> float:
        # the product distribution's entropy is the sum over its independent factors
        return float(sum(-(t[t > 0] * np.log(t[t > 0])).sum() for t in self.theta))

    def step(self, assigned: List[Tuple[int, ...]], errors: List[float], weights: List[float]) -> Dict[str, float]:
        """One exponentiated-gradient update from the round's local validation errors."""
        errors_arr = np.asarray(errors, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        baseline = _discounted_mean(self._refine_trace, self.baseline_discount) if self._refine_trace else 0.0
        refine = float(np.inner(errors_arr, w))
        self._refine_trace.append(refine)
        for i, z in enumerate(self._z):
            grad = np.zeros(len(z))
            for idx, s, wi in zip(assigned, errors_arr, w):
                grad[idx[i]] += wi * (s - baseline) / self.theta[i][idx[i]]
            if self._sched == "aggressive":
                denom = 1.0 if np.all(grad == 0.0) else float(np.abs(grad).max())
            elif self._sched == "adaptive":
                self._store[i] += float(np.abs(grad).max()) ** 2
                denom = np.sqrt(self._store[i])
            elif self._sched == "auto":
                self._store[i] += 1.0
                denom = np.sqrt(self._store[i])
            else:
                denom = 1.0
            z -= (self._eta0[i] / denom) * grad
            z -= z.max() + np.log(np.exp(z - z.max()).sum())
            self.theta[i] = np.exp(z)
        return {
            "fedex-refine-error": refine,
            "fedex-baseline": baseline,
            "fedex-entropy": self.entropy(),
            "fedex-mle-prob": float(np.prod([t.max() for t in self.theta])),
        }
