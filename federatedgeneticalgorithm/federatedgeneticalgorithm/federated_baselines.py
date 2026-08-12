"""Server-side non-GA baselines: Random Search and TPE.

Both classes implement the same interface as FederatedGA (select_for_round,
record_fitness, best_hp, best_fitness, history, generation) so they can plug
into FederatedGAFedAvg unchanged. The point is to compare the genetic algorithm
to other search strategies that share the same coupling regime (one HP per
round, broadcast to all clients, fitness = aggregated eval-acc).

Together with FederatedGA they isolate the contribution of the search
algorithm within the broadcast regime: Random Search has no memory, TPE keeps
a density model, the GA keeps an elitist population.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from federatedgeneticalgorithm.federated_genetic_algorithm import KEY_MAP


@dataclass
class FederatedRandomSearch:
    """Server-side random search: uniformly sample an HP per round, broadcast.

    Same fitness signal as FederatedGA (eval-acc post-aggregation). No
    population and no evolution: each round picks an i.i.d. sample from the
    HP grid. The `generation` attribute tracks "20-eval batches" so client
    telemetry stays compatible with the GA-shaped strategy wrapper.
    """

    hyperparams: Dict[str, List]
    pop_size: int = 4  # unused; kept for API parity with FederatedGA
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
        # Seed-individuals exhausted first, then random for the rest of the budget.
        self._seed_queue: List[Dict] = list(self.seed_individuals) if self.seed_individuals else []
        # Pre-populate `population` with a snapshot so telemetry still sees something
        # GA-shaped (server_app prints it at startup).
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
        # Bookkeeping that mirrors GA generation flips so telemetry stays uniform.
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
    """Server-side TPE (Tree-structured Parzen Estimator) via optuna.

    Each round, the optuna sampler proposes an HP given past (HP, fitness)
    pairs. We use the categorical-only space defined by HYPERPARAMS via
    explicit `CategoricalDistribution` so the search space matches GA exactly.

    `optuna.samplers.TPESampler` defaults to `n_startup_trials=10` (10 random
    trials before TPE kicks in). With a 20-round budget that's a 50/50
    explore/exploit split; keep the default to honor TPE's own design.

    seed_individuals is intentionally not implemented for TPE: it has its own
    warmup mechanism and enqueue_trial semantics don't compose cleanly with
    population seeding.
    """

    hyperparams: Dict[str, List]
    pop_size: int = 4  # unused; API parity
    seed: int = 0
    seed_individuals: Optional[List[Dict]] = None  # accepted but ignored (warned at init)

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
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._distributions = {
            KEY_MAP[k_plural]: D.CategoricalDistribution(list(values))
            for k_plural, values in self.hyperparams.items()
        }
        # Snapshot of the population for telemetry parity with FederatedGA.
        # Uses an independent RNG so it doesn't burn TPE trials before round 1.
        rng = random.Random(self.seed)
        self.population = [
            {KEY_MAP[k]: rng.choice(v) for k, v in self.hyperparams.items()}
            for _ in range(self.pop_size)
        ]
        self.fitnesses = [None] * self.pop_size
        if self.seed_individuals:
            # No-op intentionally; see class docstring.
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
