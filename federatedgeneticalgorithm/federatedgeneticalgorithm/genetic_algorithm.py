from typing import List, Dict, Tuple, Union

from deap import base, creator, tools
import numpy as np
import random

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from federatedgeneticalgorithm.task import train as train_fn
from federatedgeneticalgorithm.surrogate_model import SurrogateModel
from federatedgeneticalgorithm.config import config

# DEAP type definitions
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximization objective
if not hasattr(creator, "Individual"):
    creator.create("Individual", dict, fitness=creator.FitnessMax)

HYPERPARAMS = {
    "batch_sizes": [16, 32, 64, 128, 256, 512],
    "optimizers": ["adam", "adamw", "radam", "lion", "sgd"],
    "learning_rates": [0.0005, 0.001, 0.003, 0.005, 0.01],
    "weight_decays": [0.0, 1e-5, 1e-4, 1e-3],
    "momentums": [0.0, 0.5, 0.7, 0.9],
}

class GeneticAlgorithm:
    def __init__(self, model: nn.Module, trainset: Dataset, testset: Dataset):
        self.toolbox = base.Toolbox()
        self.model = model
        self.trainset = trainset
        self.testset = testset

        # Persist GA state across federated rounds
        self.population = []
        self.history = []
        self.elite = []
        self.round_counter = 0

        self.surrogate = SurrogateModel(HYPERPARAMS)
        self._setup_toolbox()

    def _setup_toolbox(self) -> None:
        """Register DEAP operators and keep optimizer/momentum constraints consistent."""
        self.toolbox.register("attr_batch_size", random.choice, HYPERPARAMS["batch_sizes"])
        self.toolbox.register("attr_optimizer", random.choice, HYPERPARAMS["optimizers"])
        self.toolbox.register("attr_lr", random.choice, HYPERPARAMS["learning_rates"])
        self.toolbox.register("attr_weight_decay", random.choice, HYPERPARAMS["weight_decays"])
        self.toolbox.register("attr_momentum", random.choice, HYPERPARAMS["momentums"])

        # Structure initializers
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate, indpb=config.MUTATION_PROB)
        self.toolbox.register("select", self._tournament_select, tournsize=config.TOURNAMENT_SIZE)

    def _create_individual(self) -> creator.Individual:
        """Create a random individual (hyperparameter configuration)."""
        optimizer_type = self.toolbox.attr_optimizer()

        individual = {
            "batch_size": self.toolbox.attr_batch_size(),
            "optimizer": optimizer_type,
            "lr": self.toolbox.attr_lr(),
            "weight_decay": self.toolbox.attr_weight_decay(),
        }

        if self._requires_momentum(optimizer_type):
            individual["momentum"] = self.toolbox.attr_momentum()

        return creator.Individual(individual)

    @staticmethod
    def _requires_momentum(optimizer_type: str) -> bool:
        """Return True if the optimizer uses a momentum hyperparameter."""
        return optimizer_type in ["sgd", "rmsprop"]

    def _mutate(self, individual: creator.Individual, indpb: float) -> Tuple[creator.Individual]:
        """Mutate an individual by randomly changing hyperparameters."""
        if random.random() < indpb:
            individual["batch_size"] = self.toolbox.attr_batch_size()

        if random.random() < indpb:
            individual["optimizer"] = self.toolbox.attr_optimizer()
            # Ensure momentum attribute is consistent with the new optimizer choice
            if self._requires_momentum(individual["optimizer"]):
                if "momentum" not in individual:
                    individual["momentum"] = self.toolbox.attr_momentum()
            elif "momentum" in individual:
                del individual["momentum"]

        if random.random() < indpb:
            individual["lr"] = self.toolbox.attr_lr()

        if random.random() < indpb:
            individual["weight_decay"] = self.toolbox.attr_weight_decay()

        if "momentum" in individual and random.random() < indpb:
            individual["momentum"] = self.toolbox.attr_momentum()

        return (individual,)

    def _crossover(self, ind1: creator.Individual, ind2: creator.Individual) -> Tuple[creator.Individual, creator.Individual]:
        """Perform crossover between two individuals, handling dependent attributes."""
        attributes = set(ind1.keys()) | set(ind2.keys())

        if not attributes:
            return ind1, ind2

        attribute = random.choice(list(attributes))

        if attribute == "optimizer":
            ind1["optimizer"], ind2["optimizer"] = ind2["optimizer"], ind1["optimizer"]

            for ind in [ind1, ind2]:
                if self._requires_momentum(ind["optimizer"]):
                    if "momentum" not in ind: ind["momentum"] = self.toolbox.attr_momentum()
                elif "momentum" in ind:
                    del ind["momentum"]

        elif attribute == "momentum":
            if "momentum" in ind1 and "momentum" in ind2:
                ind1["momentum"], ind2["momentum"] = ind2["momentum"], ind1["momentum"]

        elif attribute in ind1 and attribute in ind2:
            ind1[attribute], ind2[attribute] = ind2[attribute], ind1[attribute]

        return ind1, ind2

    def _tournament_select(self, population: List[creator.Individual], k: int, tournsize: int) -> List[creator.Individual]:
        """Select k individuals via tournament selection (fallback if no valid fitness is available)."""
        valid_pop = [ind for ind in population if ind.fitness.valid]
        if not valid_pop:
            return random.sample(population, k)
        return tools.selTournament(valid_pop, k, tournsize)

    def _evaluate_rung(
        self,
        individual: Union[creator.Individual, Dict],
        global_state: Dict,
        epochs: int,
        subset_ratio: float,
        mu: float,
    ) -> float:
        """Evaluate HPs at a given fidelity (epochs/subset); returns accuracy."""
        self.model.load_state_dict(global_state)

        dataset_len = len(self.trainset)
        subset_size = max(1, int(dataset_len * subset_ratio))
        indices = np.random.choice(dataset_len, subset_size, replace=False)
        subset = Subset(self.trainset, indices)

        loader = DataLoader(
            subset,
            batch_size=individual["batch_size"],
            shuffle=True,
            num_workers=0
        )

        metrics = train_fn(
            self.model,
            loader,
            epochs=epochs,
            lr=individual["lr"],
            device=config.DEVICE,
            optimizer=individual["optimizer"],
            weight_decay=individual["weight_decay"],
            momentum=individual.get("momentum", 0.0),
            mu=mu, # Controls FedProx regularization strength
            global_state_dict=global_state
        )
        return metrics["accuracy"]

    def run_round_updates(self, global_state_dict: Dict) -> Dict:
        """Run one incremental GA iteration: generate candidates, evaluate, and update elite/surrogate."""
        self.round_counter += 1
        candidates_pool = []

        # 1. Generate candidate pool (Elite + Offspring)
        # Warm start from elite when available
        if not self.population:
            if self.elite:
                for elite_entry in self.elite[: min(3, len(self.elite))]:
                    candidates_pool.append(creator.Individual(elite_entry["hp"]))
                remaining = config.POPULATION_SIZE - len(candidates_pool)
                candidates_pool.extend(self.toolbox.population(n=remaining))
                self.population = candidates_pool
            else:
                self.population = self.toolbox.population(n=config.POPULATION_SIZE)

        offspring_size = max(3, config.POPULATION_SIZE // 2)
        offspring = self.toolbox.select(self.population, offspring_size)
        offspring = list(map(self.toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config.CROSSOVER_PROB:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < config.MUTATION_PROB:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        candidates_pool.extend(offspring)

        if self.elite:
            for elite_entry in self.elite[:2]:
                elite_ind = creator.Individual(elite_entry["hp"])
                elite_ind.fitness.values = (elite_entry["fitness"],)
                candidates_pool.append(elite_ind)

        # 2. Inject candidates predicted by Surrogate Model (if trained)
        if self.surrogate.ready:
            trials = [self._create_individual() for _ in range(30)]
            scores = self.surrogate.predict_batch(trials)
            top_indices = np.argsort(scores)[-2:]
            for idx in top_indices:
                candidates_pool.append(trials[idx])

        self.population[:] = candidates_pool[: config.POPULATION_SIZE]

        # 3. Low-fidelity evaluation (Rung 0) on a subset of candidates
        num_to_eval = min(len(candidates_pool), config.NUM_CANDIDATES_TO_EVALUATE)
        selection_to_eval = random.sample(candidates_pool, k=num_to_eval)
        rung0_results = []

        for ind in selection_to_eval:
            acc = self._evaluate_rung(
                ind,
                global_state_dict,
                config.RUNG0_EPOCHS,
                config.RUNG0_SUBSET_RATIO,
                config.RUNG0_MU,
            )
            ind.fitness.values = (acc,)

            entry = {"hp": dict(ind), "fitness": acc, "rung": 0}
            self.history.append(entry)
            rung0_results.append(entry)

        # 4. High-fidelity evaluation (Rung 1) used to promote the best candidate
        if rung0_results:
            best_rung0 = max(rung0_results, key=lambda x: x["fitness"])

            acc_r1 = self._evaluate_rung(
                best_rung0["hp"],
                global_state_dict,
                config.RUNG1_EPOCHS,
                1.0,
                config.RUNG1_MU,
            )

            entry_r1 = {"hp": best_rung0["hp"], "fitness": acc_r1, "rung": 1}
            self.history.append(entry_r1)

            self.elite.append(entry_r1)
            self.elite.sort(key=lambda x: x["fitness"], reverse=True)
            self.elite = self.elite[:3]

        # 5. Periodically retrain the surrogate model using accumulated history
        if self.round_counter % config.SURROGATE_RETRAIN_INTERVAL == 0 and len(self.history) >= 5:
            self.surrogate.update(self.history)

        if self.elite:
            return self.elite[0]["hp"]
        elif rung0_results:
            best_current = max(rung0_results, key=lambda x: x["fitness"])
            return best_current["hp"]
        else:
            return selection_to_eval[0]
