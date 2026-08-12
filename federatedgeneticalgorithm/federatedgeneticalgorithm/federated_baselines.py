"""Random Search and TPE with the FederatedGA interface, so FederatedGAFedAvg can run them as is."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
