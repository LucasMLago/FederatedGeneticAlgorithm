"""Server-side federated GA: picks one HP per round, broadcast to all clients.

Ablation against per-client GA: instead of each client running its own GA over
its private val set, the server maintains a single population and chooses one
HP per FL round. All clients train with that HP, FedAvg aggregates, and the
server evaluates the aggregated model -- its eval-acc becomes that HP's fitness.

This removes inter-client HP heterogeneity (the failure mode identified in
Exps B/D where mixed optimizers/LRs hurt FedAvg aggregation).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Mapping between HYPERPARAMS dict keys (plural) and per-HP dict keys (singular)
KEY_MAP = {
    "batch_sizes": "batch_size",
    "optimizers": "optimizer",
    "learning_rates": "lr",
    "weight_decays": "weight_decay",
    "momentums": "momentum",
}


@dataclass
class FederatedGA:
    """Server-side GA: one HP evaluated per FL round, fitness = aggregated eval-acc."""

    hyperparams: Dict[str, List]
    pop_size: int = 4
    mutation_prob: float = 0.3
    crossover_prob: float = 0.7
    tournament_size: int = 2
    seed: int = 0
    seed_individuals: Optional[List[Dict]] = None  # hand-picked HPs to put in initial pop

    population: List[Dict] = field(default_factory=list)
    fitnesses: List[Optional[float]] = field(default_factory=list)
    history: List[Tuple[int, Dict, float]] = field(default_factory=list)  # (gen, hp, fitness)
    current_idx: int = 0
    generation: int = 0
    best_hp: Optional[Dict] = None
    best_fitness: float = float("-inf")
    _rng: random.Random = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if not self.population:
            seeded = list(self.seed_individuals) if self.seed_individuals else []
            seeded = seeded[: self.pop_size]
            random_fill = [self._random_hp() for _ in range(self.pop_size - len(seeded))]
            self.population = seeded + random_fill
        if not self.fitnesses:
            self.fitnesses = [None] * self.pop_size

    def _random_hp(self) -> Dict:
        return {
            KEY_MAP[k_plural]: self._rng.choice(values)
            for k_plural, values in self.hyperparams.items()
        }

    def select_for_round(self, server_round: int) -> Dict:
        """Return the HP to be broadcast for this round (round-robin within generation)."""
        return dict(self.population[self.current_idx])

    def record_fitness(self, fitness: float) -> Dict[str, object]:
        """Record fitness for the HP just used; evolve if generation complete."""
        idx = self.current_idx
        hp = dict(self.population[idx])
        self.fitnesses[idx] = fitness
        self.history.append((self.generation, hp, fitness))
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_hp = hp
        info = {
            "generation": self.generation,
            "individual": idx,
            "hp": hp,
            "fitness": fitness,
            "is_new_best": (self.best_hp is not None and self.best_hp == hp and fitness == self.best_fitness),
        }
        self.current_idx = (self.current_idx + 1) % self.pop_size
        if self.current_idx == 0 and all(f is not None for f in self.fitnesses):
            self._evolve()
            info["evolved_to_generation"] = self.generation
        return info

    def _tournament(self) -> Dict:
        contenders = self._rng.sample(range(self.pop_size), self.tournament_size)
        scored = [(self.fitnesses[i] if self.fitnesses[i] is not None else float("-inf"), i) for i in contenders]
        scored.sort(reverse=True)
        return dict(self.population[scored[0][1]])

    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        return {k: (p1[k] if self._rng.random() < 0.5 else p2[k]) for k in p1}

    def _mutate(self, hp: Dict) -> Dict:
        out = dict(hp)
        for k_plural, values in self.hyperparams.items():
            singular = KEY_MAP[k_plural]
            if self._rng.random() < self.mutation_prob:
                out[singular] = self._rng.choice(values)
        return out

    def _evolve(self) -> None:
        self.generation += 1
        new_pop: List[Dict] = []
        # Elitism: best HP carried over
        if self.best_hp is not None:
            new_pop.append(dict(self.best_hp))
        while len(new_pop) < self.pop_size:
            p1 = self._tournament()
            p2 = self._tournament()
            child = self._crossover(p1, p2) if self._rng.random() < self.crossover_prob else dict(p1)
            child = self._mutate(child)
            new_pop.append(child)
        self.population = new_pop
        self.fitnesses = [None] * self.pop_size
