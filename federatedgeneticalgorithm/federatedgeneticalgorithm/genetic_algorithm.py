import random
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
from deap import base, creator, tools, algorithms
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import multiprocessing
from multiprocessing.pool import ThreadPool

import numpy as np

import sys
sys.path.append("../")
from config import config

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)

HYPERPARAMS = {
    "batch_sizes": [16, 32, 64, 128, 256, 512, 1024],
    "optimizers": ["sgd", "adam", "rmsprop"],
    "learning_rates": [0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
    "weight_decays": [0.0, 1e-5, 1e-4, 1e-3],
    "momentums": [0.0, 0.3, 0.5, 0.7, 0.9, 0.99],
}


class GeneticAlgorithm:
    def __init__(self, model: nn.Module, trainset: Dataset, testset: Dataset):
        self.toolbox = base.Toolbox()
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self._setup_toolbox()

    def _setup_toolbox(self) -> None:
        """Configure DEAP toolbox with genetic operators."""
        # Attribute generators
        self.toolbox.register("attr_batch_size", random.choice, HYPERPARAMS["batch_sizes"])
        self.toolbox.register("attr_optimizer", random.choice, HYPERPARAMS["optimizers"])
        self.toolbox.register("attr_lr", random.choice, HYPERPARAMS["learning_rates"])
        self.toolbox.register("attr_weight_decay", random.choice, HYPERPARAMS["weight_decays"])
        self.toolbox.register("attr_momentum", random.choice, HYPERPARAMS["momentums"])

        # Structure initializers
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate, indpb=config.MUTATION_PROB)
        self.toolbox.register("select", self._tournament_select, tournsize=3)
    
    def build_dataloaders(self, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Build train, validation and test dataloaders with consistent seeding."""
        train_size = int(0.8 * len(self.trainset))
        val_size = len(self.trainset) - train_size

        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(self.trainset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, generator=generator)
        test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, generator=generator)

        return train_loader, val_loader, test_loader

    def _create_individual(self) -> Dict:
        """Create a random individual (hyperparameter configuration)."""
        optimizer_type = self.toolbox.attr_optimizer()
        
        # Create a copy of the model for each individual
        model_copy = copy.deepcopy(self.model)
        
        individual = {
            "model": model_copy,
            "batch_size": self.toolbox.attr_batch_size(),
            "optimizer": optimizer_type,
            "lr": self.toolbox.attr_lr(),
            "weight_decay": self.toolbox.attr_weight_decay(),
        }
        
        # Add momentum only for optimizers that support it
        if self._requires_momentum(optimizer_type):
            individual["momentum"] = self.toolbox.attr_momentum()
            
        return individual
    
    @staticmethod
    def _requires_momentum(optimizer_type: str) -> bool:
        """Check if optimizer requires momentum parameter."""
        return optimizer_type in ("sgd", "rmsprop")
    
    def _create_optimizer(
        self, 
        model: nn.Module, 
        optimizer_type: str, 
        lr: float, 
        weight_decay: float, 
        momentum: Optional[float] = None
    ) -> optim.Optimizer:
        """Factory method to create optimizer instance."""
        params = model.parameters()
        
        if optimizer_type == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "rmsprop":
            return optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum or 0.0)
        elif optimizer_type == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum or 0.0)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    @staticmethod
    def _calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate classification accuracy."""
        predicted = torch.argmax(logits, dim=1).cpu()
        return (predicted.eq(labels).sum().item()) / labels.shape[0]

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model on given DataLoader. Returns average loss and accuracy."""
        model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        
        for x, y in loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            accuracy = self._calculate_accuracy(logits, y.cpu())

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy * batch_size
            n += batch_size
            
        return total_loss / n, total_acc / n
    
    def _evaluate_individual(self, individual: creator.Individual) -> Tuple[float]:
        """Train and evaluate a model with given hyperparameters."""
        # Extract hyperparameters
        model: nn.Module = individual["model"]
        batch_size: int = individual["batch_size"]
        optimizer_type: str = individual["optimizer"]
        lr: float = individual["lr"]
        weight_decay: float = individual["weight_decay"]
        momentum: float = individual.get("momentum")

        # Reset model to initial state before training
        model.load_state_dict(self.model.state_dict())
        model.to(config.DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = self._create_optimizer(model, optimizer_type, lr, weight_decay, momentum)
        train_loader, val_loader, _ = self.build_dataloaders(batch_size, config.SEED)

        # Training loop with early stopping
        best_val_acc = 0.0
        patience_counter = 0
        patience = 5
        
        for epoch in range(config.EPOCHS):
            # Training phase
            model.train()
            for x, y in train_loader:
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            # Validation
            val_loss, val_acc = self.evaluate(model, val_loader, criterion)
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return (best_val_acc,)
    
    def _mutate(self, individual: creator.Individual, indpb: float) -> Tuple[creator.Individual]:
        """Mutate individual by randomly changing hyperparameters."""
        if random.random() < indpb:
            individual["batch_size"] = self.toolbox.attr_batch_size()
        if random.random() < indpb:
            individual["optimizer"] = self.toolbox.attr_optimizer()
        if random.random() < indpb:
            individual["lr"] = self.toolbox.attr_lr()
        if random.random() < indpb:
            individual["weight_decay"] = self.toolbox.attr_weight_decay()
        
        # Mutate momentum only for compatible optimizers
        if self._requires_momentum(individual["optimizer"]):
            if random.random() < indpb:
                individual["momentum"] = self.toolbox.attr_momentum()
        elif "momentum" in individual:
            del individual["momentum"]  # Remove momentum if optimizer changed

        return (individual,)

    def _crossover(self, ind1: creator.Individual, ind2: creator.Individual) -> Tuple[creator.Individual, creator.Individual]:
        """Perform crossover between two individuals."""
        attributes = [key for key in ind1.keys() if key != "model"]
        
        # Filter valid attributes for crossover
        valid_attrs = [
            attr for attr in attributes 
            if attr != "momentum" or (
                self._requires_momentum(ind1["optimizer"]) and 
                self._requires_momentum(ind2["optimizer"])
            )
        ]

        if valid_attrs:
            attribute = random.choice(valid_attrs)
            ind1[attribute], ind2[attribute] = ind2[attribute], ind1[attribute]

        return ind1, ind2

    def _tournament_select(self, population: List[creator.Individual], k: int, tournsize: int) -> List[creator.Individual]:
        """Select k individuals using tournament selection."""
        selected = []
        for _ in range(k):
            aspirants = random.sample(population, tournsize)
            best = max(aspirants, key=lambda ind: ind.fitness.values[0])
            selected.append(best)
        return selected
    
    def run(self) -> Tuple[List[creator.Individual], tools.Logbook]:
        """Execute the genetic algorithm optimization process."""
        # Setup parallel evaluation
        pool = ThreadPool(multiprocessing.cpu_count())
        self.toolbox.register("map", pool.map)
        
        # Initialize population and statistics
        population = self.toolbox.population(n=config.POPULATION_SIZE)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run evolution
        try:
            population, logbook = algorithms.eaSimple(
                population,
                self.toolbox,
                cxpb=config.CROSSOVER_PROB,
                mutpb=config.MUTATION_PROB,
                ngen=config.NUMBER_OF_GENERATIONS,
                stats=stats,
                halloffame=hof,
                verbose=False,
            )
        finally:
            pool.close()
            pool.join()

        return population, logbook
    
    def get_best_individuals(self, population: List[creator.Individual], k: int) -> List[creator.Individual]:
        """Get top k individuals from population."""
        return tools.selBest(population, k)
