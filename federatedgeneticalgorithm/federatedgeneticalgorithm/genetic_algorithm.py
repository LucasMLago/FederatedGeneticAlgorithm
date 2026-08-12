from typing import List, Dict, Tuple, Union, Optional

import fcntl
import gc
import os
import pickle
import tempfile

from deap import base, creator, tools
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from federatedgeneticalgorithm.task import train as train_fn
from federatedgeneticalgorithm.task import test as test_fn
from federatedgeneticalgorithm.surrogate_model import SurrogateModel
from federatedgeneticalgorithm.config import config
from federatedgeneticalgorithm import telemetry

# DEAP type definitions
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximization objective
if not hasattr(creator, "Individual"):
    creator.create("Individual", dict, fitness=creator.FitnessMax)

HYPERPARAMS = {
    "batch_sizes": [64, 128], # Restringido para evitar Client Drift severo no FedAvg
    "optimizers": ["adam", "adamw", "radam", "lion", "sgd"],
    "learning_rates": [0.0005, 0.001, 0.003, 0.005, 0.01], # Exp D: 0.01 reincluido para igualar baseline
    "weight_decays": [0.0, 1e-5, 1e-4, 1e-3],
    "momentums": [0.5, 0.7, 0.9, 0.95],
}


def read_shared_pool(path: str) -> List[Dict]:
    """Read the federated surrogate training pool with a shared lock."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return pickle.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (pickle.UnpicklingError, EOFError, OSError):
        return []


def append_shared_pool(path: str, new_entries: List[Dict]) -> None:
    """Atomic read-modify-write append to the federated surrogate pool."""
    if not new_entries:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.lseek(fd, 0, os.SEEK_SET)
        data = b""
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            data += chunk
        try:
            pool = pickle.loads(data) if data else []
        except (pickle.UnpicklingError, EOFError):
            pool = []
        pool.extend(new_entries)
        serialized = pickle.dumps(pool, protocol=pickle.HIGHEST_PROTOCOL)
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, serialized)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

class GeneticAlgorithm:
    def __init__(self, model: nn.Module, trainset: Dataset, testset: Dataset):
        self.toolbox = base.Toolbox()
        self.model = model
        self.trainset = trainset
        self.testset = testset

        # State kept across federated rounds (persisted via save_state/load_state).
        self.population = []
        self.history = []
        self.elite = []
        self.round_counter = 0
        self.visit_counter = 0
        self.best_round_fitness: List[float] = []
        # (hp_sig, rung) -> (round_evaluated, payload); used to skip re-training
        # a HP whose evaluation is still fresh (see FITNESS_CACHE_MAX_ROUND_AGE).
        self.fitness_cache: Dict[Tuple, Tuple[int, float]] = {}
        self.last_best_hp: Optional[Dict] = None
        self.last_best_fitness: float = 0.0

        self.surrogate = SurrogateModel(HYPERPARAMS)
        self._setup_toolbox()

    # ---- Disk persistence (Ray's actor pool recycles workers) ----------------

    def save_state(self, path: str) -> None:
        """Atomically serialize GA state to `path`."""
        state = {
            "round_counter": int(self.round_counter),
            "visit_counter": int(self.visit_counter),
            "best_round_fitness": list(self.best_round_fitness),
            "history": [dict(e) for e in self.history],
            "elite": [dict(e) for e in self.elite],
            "population": [dict(ind) for ind in self.population],
            "fitness_cache": dict(self.fitness_cache),
            "last_best_hp": dict(self.last_best_hp) if self.last_best_hp else None,
            "last_best_fitness": float(self.last_best_fitness),
            "surrogate_model": self.surrogate.model,
            "surrogate_ready": bool(self.surrogate.ready),
        }
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".ga_state_", dir=directory)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def load_state(self, path: str) -> bool:
        """Rehydrate GA state from `path`. Returns True if loaded."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError):
            return False

        self.round_counter = int(state.get("round_counter", 0))
        self.visit_counter = int(state.get("visit_counter", 0))
        self.best_round_fitness = list(state.get("best_round_fitness", []))
        self.history = list(state.get("history", []))
        self.elite = list(state.get("elite", []))
        self.population = [creator.Individual(d) for d in state.get("population", [])]
        self.fitness_cache = dict(state.get("fitness_cache", {}))
        self.last_best_hp = state.get("last_best_hp")
        self.last_best_fitness = float(state.get("last_best_fitness", 0.0))
        if state.get("surrogate_model") is not None:
            self.surrogate.model = state["surrogate_model"]
        self.surrogate.ready = bool(state.get("surrogate_ready", False))
        return True

    @staticmethod
    def _hp_signature(individual: Union[creator.Individual, Dict]) -> Tuple:
        hp = dict(individual)
        return (
            hp.get("batch_size"),
            hp.get("optimizer"),
            hp.get("lr"),
            hp.get("weight_decay"),
            hp.get("momentum", 0.0),
        )

    def _is_stagnating(self) -> bool:
        window = max(2, int(config.GA_STAGNATION_WINDOW))
        if len(self.best_round_fitness) < window:
            return False
        recent = self.best_round_fitness[-window:]
        gain = max(recent) - min(recent)
        return gain < float(config.GA_STAGNATION_MIN_DELTA)

    def _sample_shared_indices(self, subset_ratio: float) -> np.ndarray:
        dataset_len = len(self.trainset)
        subset_size = max(1, int(dataset_len * subset_ratio))
        seed = config.SEED + (self.round_counter * 997)  # round-stable and client-specific
        rng = np.random.default_rng(seed)
        return rng.choice(dataset_len, subset_size, replace=False)

    def _high_fidelity_history(self) -> List[Dict]:
        return [
            entry
            for entry in self.history
            if entry.get("rung") in {1, "elite-reval", "full"}
        ]

    @staticmethod
    def _shared_pool_path() -> str:
        """Run-scoped file where all clients pool their high-fidelity (hp, fitness) samples."""
        return str(telemetry.get_run_dir() / "shared_hp_pool.pkl")

    def _surrogate_guided_candidates(self, pool_size: int, top_k: int, stagnating: bool) -> List[creator.Individual]:
        if not self.surrogate.ready:
            return []

        trials = [self._create_individual() for _ in range(max(top_k, pool_size))]
        scored = self.surrogate.predict_with_uncertainty(trials)
        kappa = 0.60 if stagnating else 0.35
        ranked = sorted(
            zip(trials, scored),
            key=lambda t: (t[1][0] + kappa * t[1][1]),
            reverse=True,
        )
        return [ind for ind, _ in ranked[:top_k]]

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
        """Select k individuals via tournament selection. Falls back to sampling
        with replacement when population is smaller than k (happens when state is
        restored from disk with a partial population, or when stagnation widens
        offspring_size beyond POPULATION_SIZE)."""
        if not population:
            return [self._create_individual() for _ in range(k)]
        valid_pop = [ind for ind in population if ind.fitness.valid]
        if not valid_pop:
            # No fitness signal yet -- uniform sampling with replacement.
            return [random.choice(population) for _ in range(k)]
        if len(valid_pop) < k:
            # Tournament needs enough distinct candidates per draw; sample with
            # replacement to reach k without raising.
            return [random.choice(valid_pop) for _ in range(k)]
        return tools.selTournament(valid_pop, k, tournsize)

    def _cache_get(self, sig: Tuple, rung_tag: str) -> Optional[Tuple[float, float, float]]:
        key = (sig, rung_tag)
        entry = self.fitness_cache.get(key)
        if entry is None:
            return None
        cached_round, payload = entry
        if self.round_counter - cached_round > config.FITNESS_CACHE_MAX_ROUND_AGE:
            return None
        # Backward-compat: legacy caches stored just fitness; promote to a triple
        # so callers can always unpack (fitness, drift, val_acc). The drift/val_acc
        # are unknown for legacy entries; mark them as 0/fitness so downstream
        # arithmetic degrades gracefully.
        if isinstance(payload, tuple) and len(payload) == 3:
            return payload
        return (float(payload), 0.0, float(payload))

    def _cache_put(self, sig: Tuple, rung_tag: str, payload: Tuple[float, float, float]) -> None:
        self.fitness_cache[(sig, rung_tag)] = (self.round_counter, payload)

    def _history_entry(
        self,
        hp: Dict,
        fitness: float,
        drift: float,
        val_acc: float,
        rung,
    ) -> Dict:
        """Build a history/pool entry carrying drift and a post-agg proxy.

        post_agg_proxy = val_acc - lambda * drift. Same shape as `fitness`, but
        kept separate so callers can pick which target the surrogate learns.
        """
        lam = self._current_fitness_lambda()
        post_agg_proxy = max(0.0, float(val_acc) - lam * float(drift))
        return {
            "hp": hp,
            "fitness": float(fitness),
            "drift": float(drift),
            "val_acc": float(val_acc),
            "post_agg_proxy": float(post_agg_proxy),
            "rung": rung,
        }

    def _current_fitness_lambda(self) -> float:
        """Lambda ramped linearly from FITNESS_LAMBDA_WARMUP_START to the target.

        Early rounds see large drift (the global is untrained), so a full
        penalty would kill aggressive HPs that are actually making progress.
        """
        target = float(getattr(config, "FITNESS_DRIFT_PENALTY_LAMBDA", 0.0))
        warmup = int(getattr(config, "FITNESS_LAMBDA_WARMUP_ROUNDS", 0))
        start = float(getattr(config, "FITNESS_LAMBDA_WARMUP_START", target))
        if warmup <= 0 or self.round_counter >= warmup:
            return target
        progress = max(0.0, min(1.0, self.round_counter / warmup))
        return start + (target - start) * progress

    def _evaluate_rung(
        self,
        individual: Union[creator.Individual, Dict],
        global_state: Dict,
        epochs: int,
        subset_ratio: float,
        mu: float,
        shared_indices: Optional[np.ndarray] = None,
        rung_tag: str = "rung0",
    ) -> Optional[Tuple[float, float, float]]:
        """Train+eval this HP at the given fidelity. Returns (fitness, drift, val_acc).

        On CUDA OOM returns None so the candidate is dropped entirely rather
        than poisoning the surrogate with a fake low fitness.
        """
        sig = self._hp_signature(individual)
        cached = self._cache_get(sig, rung_tag)
        if cached is not None:
            return cached

        self.model.load_state_dict(global_state)

        dataset_len = len(self.trainset)
        if shared_indices is not None:
            indices = shared_indices
        else:
            subset_size = max(1, int(dataset_len * subset_ratio))
            indices = np.random.choice(dataset_len, subset_size, replace=False)
        subset = Subset(self.trainset, indices)

        train_size = max(1, int(0.8 * len(subset)))
        val_size = len(subset) - train_size
        gen = torch.Generator().manual_seed(config.SEED + self.round_counter)
        train_subset, val_subset = random_split(subset, [train_size, val_size], generator=gen)

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            train_subset,
            batch_size=individual["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=pin_memory,
            generator=gen,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=individual["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=pin_memory,
            generator=gen,
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            train_fn(
                self.model,
                train_loader,
                epochs=epochs,
                lr=individual["lr"],
                device=device,
                optimizer=individual["optimizer"],
                weight_decay=individual["weight_decay"],
                momentum=individual.get("momentum", 0.0),
                mu=mu,  # Controls FedProx regularization strength
                global_state_dict=global_state
            )
            _, val_acc = test_fn(self.model, val_loader, device)
            drift = self._compute_drift(global_state, device)
        except torch.cuda.OutOfMemoryError:
            print(
                f"[GA OOM] round={self.round_counter} rung={rung_tag} hp={dict(individual)} -- candidate dropped",
                flush=True,
            )
            val_acc = None
            drift = 0.0

        del train_loader, val_loader, train_subset, val_subset, subset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if val_acc is not None:
            lam = self._current_fitness_lambda()
            fitness = max(0.0, val_acc - lam * drift) if lam > 0 else float(val_acc)
            result = (float(fitness), float(drift), float(val_acc))
            self._cache_put(sig, rung_tag, result)
            return result
        return None

    def _compute_drift(self, global_state: Dict, device: torch.device) -> float:
        """RMS of (local - global) across parameters. Size-invariant."""
        with torch.no_grad():
            total_sq = 0.0
            total_n = 0
            for name, param in self.model.named_parameters():
                if name not in global_state:
                    continue
                gp = global_state[name].to(device, non_blocking=True)
                diff = param.detach() - gp
                total_sq += float(diff.pow(2).sum().item())
                total_n += diff.numel()
            if total_n == 0:
                return 0.0
            return (total_sq / total_n) ** 0.5

    def run_round_updates(self, global_state_dict: Dict, client_id: int) -> Dict:
        """Run one incremental GA iteration: generate candidates, evaluate, and update elite/surrogate."""
        history_start_idx = len(self.history)
        self.round_counter += 1
        candidates_pool = []
        stagnating = self._is_stagnating()

        def _append_unique(individual: creator.Individual) -> None:
            sig = self._hp_signature(individual)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                candidates_pool.append(individual)

        # 1. Generate candidate pool (Elite + Offspring)
        # Warm start from elite when available
        seen_signatures = set()
        if not self.population:
            if self.elite:
                for elite_entry in self.elite[: min(3, len(self.elite))]:
                    _append_unique(creator.Individual(elite_entry["hp"]))
                remaining = config.POPULATION_SIZE - len(candidates_pool)
                for ind in self.toolbox.population(n=max(0, remaining)):
                    _append_unique(ind)
                self.population = candidates_pool
            else:
                self.population = self.toolbox.population(n=config.POPULATION_SIZE)
                for ind in self.population:
                    _append_unique(ind)
        else:
            for ind in self.population:
                _append_unique(ind)

        offspring_size = max(3, config.POPULATION_SIZE // 2) + (config.ADAPTIVE_EXTRA_MUTANTS if stagnating else 0)
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

        for child in offspring:
            _append_unique(child)

        if self.elite:
            for elite_entry in self.elite[:2]:
                elite_ind = creator.Individual(elite_entry["hp"])
                elite_ind.fitness.values = (elite_entry["fitness"],)
                _append_unique(elite_ind)

        # OPTIMIZED GA WITH SURROGATE + MULTI-FIDELITY
        if config.ENABLE_SURROGATE_GA:
            # 2. Inject candidates predicted by Surrogate Model (if trained)
            guided = self._surrogate_guided_candidates(
                pool_size=config.SURROGATE_TRIAL_POOL_SIZE,
                top_k=config.SURROGATE_TOP_K + (1 if stagnating else 0),
                stagnating=stagnating,
            )
            for ind in guided:
                _append_unique(ind)

            self.population[:] = candidates_pool[: config.POPULATION_SIZE]

            # 3. Rung 0: surrogate-first when trained; otherwise real low-fidelity training.
            # Once the surrogate is ready and we're past the warmup, use its prediction as
            # fitness for most candidates and spend real compute only on 2 picks per round:
            # one by expected improvement (exploit) and one by uncertainty (explore) to keep
            # the surrogate calibrated. The Rung 1 promotion still does real training, so the
            # final selected HP is always validated with actual data.
            kappa = 0.60 if stagnating else 0.35
            surrogate_first = (
                self.surrogate.ready
                and self.round_counter >= config.SURROGATE_FIRST_WARMUP_ROUNDS
            )

            rung0_results = []
            rung1_results = []

            shared_rung0_indices = None
            if config.RUNG0_USE_SHARED_SUBSET and not surrogate_first:
                shared_rung0_indices = self._sample_shared_indices(config.RUNG0_SUBSET_RATIO)

            if surrogate_first:
                scored_pool = self.surrogate.predict_with_uncertainty(candidates_pool)
                scored = list(zip(candidates_pool, scored_pool))

                by_ei = sorted(scored, key=lambda t: t[1][0] + kappa * t[1][1], reverse=True)
                real_evals = [by_ei[0][0]]

                n_real = min(
                    max(1, config.SURROGATE_FIRST_REAL_EVALS_PER_ROUND),
                    len(candidates_pool),
                )
                if n_real > 1:
                    remaining = [t for t in scored if t[0] is not real_evals[0]]
                    by_uncertainty = sorted(remaining, key=lambda t: t[1][1], reverse=True)
                    real_evals += [ind for ind, _ in by_uncertainty[: n_real - 1]]

                real_ids = {id(ind) for ind in real_evals}

                for ind in real_evals:
                    result = self._evaluate_rung(
                        ind,
                        global_state_dict,
                        config.RUNG0_EPOCHS,
                        config.RUNG0_SUBSET_RATIO,
                        config.RUNG0_MU,
                        shared_indices=shared_rung0_indices,
                        rung_tag="rung0",
                    )
                    if result is None:
                        continue
                    fitness, drift, val_acc = result
                    ind.fitness.values = (fitness,)
                    entry = self._history_entry(dict(ind), fitness, drift, val_acc, rung=0)
                    self.history.append(entry)
                    rung0_results.append(entry)

                # Surrogate-only candidates: fitness comes from the model prediction. Do
                # NOT append to history to avoid training the surrogate on its own output.
                for ind, (mean_pred, _std) in scored:
                    if id(ind) in real_ids:
                        continue
                    ind.fitness.values = (float(mean_pred),)
                    rung0_results.append(
                        {
                            "hp": dict(ind),
                            "fitness": float(mean_pred),
                            "rung": "surrogate",
                            "drift": 0.0,
                            "val_acc": float(mean_pred),
                            "post_agg_proxy": float(mean_pred),
                        }
                    )
            else:
                num_to_eval = min(len(candidates_pool), config.NUM_CANDIDATES_TO_EVALUATE)
                if self.surrogate.ready:
                    scored_pool = self.surrogate.predict_with_uncertainty(candidates_pool)
                    ranked_pool = sorted(
                        zip(candidates_pool, scored_pool),
                        key=lambda t: (t[1][0] + kappa * t[1][1]),
                        reverse=True,
                    )
                    selection_to_eval = [ind for ind, _ in ranked_pool[:num_to_eval]]
                else:
                    selection_to_eval = random.sample(candidates_pool, k=num_to_eval)

                for ind in selection_to_eval:
                    result = self._evaluate_rung(
                        ind,
                        global_state_dict,
                        config.RUNG0_EPOCHS,
                        config.RUNG0_SUBSET_RATIO,
                        config.RUNG0_MU,
                        shared_indices=shared_rung0_indices,
                        rung_tag="rung0",
                    )
                    if result is None:
                        continue
                    fitness, drift, val_acc = result
                    ind.fitness.values = (fitness,)

                    entry = self._history_entry(dict(ind), fitness, drift, val_acc, rung=0)
                    self.history.append(entry)
                    rung0_results.append(entry)

            # 4. High-fidelity evaluation (Rung 1) used to promote the best candidates
            if rung0_results:
                top_k = max(1, min(config.RUNG1_TOP_K, len(rung0_results)))
                promoted = sorted(rung0_results, key=lambda x: x["fitness"], reverse=True)[:top_k]

                for candidate in promoted:
                    result_r1 = self._evaluate_rung(
                        candidate["hp"],
                        global_state_dict,
                        config.RUNG1_EPOCHS,
                        1.0,  # Full dataset
                        config.RUNG1_MU,
                        rung_tag="rung1",
                    )
                    if result_r1 is None:
                        continue
                    fit_r1, drift_r1, val_r1 = result_r1
                    entry_r1 = self._history_entry(candidate["hp"], fit_r1, drift_r1, val_r1, rung=1)
                    self.history.append(entry_r1)
                    rung1_results.append(entry_r1)

            # Revalidate historical elite under the current global state to avoid stale picks.
            # Gated by ELITE_REEVAL_EVERY_N_ROUNDS: high-fidelity re-eval is expensive and
            # the elite fitness is only mildly stale across a few rounds.
            elite_revalidated = []
            should_reeval_elite = (
                self.elite
                and self.round_counter % max(1, config.ELITE_REEVAL_EVERY_N_ROUNDS) == 0
            )
            if should_reeval_elite:
                reevaluate_k = max(1, min(config.ELITE_REEVAL_COUNT, len(self.elite)))
                elite_candidates = sorted(self.elite, key=lambda x: x["fitness"], reverse=True)[:reevaluate_k]
                for elite_entry in elite_candidates:
                    elite_result = self._evaluate_rung(
                        elite_entry["hp"],
                        global_state_dict,
                        config.RUNG1_EPOCHS,
                        1.0,
                        config.RUNG1_MU,
                        rung_tag="rung1",
                    )
                    if elite_result is None:
                        continue
                    fit_el, drift_el, val_el = elite_result
                    refreshed = self._history_entry(
                        dict(elite_entry["hp"]), fit_el, drift_el, val_el, rung="elite-reval"
                    )
                    self.history.append(refreshed)
                    elite_revalidated.append(refreshed)

            # Keep a compact elite set with fresh (current-round) fitness values.
            fresh_pool = rung1_results + elite_revalidated
            if fresh_pool:
                self.elite = sorted(fresh_pool, key=lambda x: x["fitness"], reverse=True)[:3]

            # 5. Federated pool: share this round's new high-fidelity samples with
            # other clients via disk, and retrain the surrogate on the pooled data.
            # Pooling is what makes the surrogate usable at all: one client alone
            # collects 4-5 high-fidelity samples per run, while the shared pool
            # reaches tens to hundreds several rounds earlier.
            pool_path = self._shared_pool_path()
            new_hf_entries = [
                {
                    "hp": dict(e["hp"]),
                    "fitness": float(e["fitness"]),
                    "drift": float(e.get("drift", 0.0)),
                    "val_acc": float(e.get("val_acc", e["fitness"])),
                    "post_agg_proxy": float(e.get("post_agg_proxy", e["fitness"])),
                    "client_id": int(client_id),
                    "round": int(self.round_counter),
                }
                for e in self.history[history_start_idx:]
                if e.get("rung") in {1, "elite-reval"}
            ]
            append_shared_pool(pool_path, new_hf_entries)
            pooled_training_data = read_shared_pool(pool_path)

            if (
                self.round_counter % config.SURROGATE_RETRAIN_INTERVAL == 0
                and len(pooled_training_data) >= config.SURROGATE_MIN_HIGH_FIDELITY_SAMPLES
            ):
                self.surrogate.update(pooled_training_data)

            # The client is represented by the best individual in the current global state.
            selection_pool = rung1_results + elite_revalidated
            if not selection_pool:
                selection_pool = rung0_results

            if selection_pool:
                best_current = max(selection_pool, key=lambda x: x["fitness"])
                self.best_round_fitness.append(float(best_current["fitness"]))
                telemetry.append_ga_candidates_rows(
                    client_id=client_id,
                    client_round=self.round_counter,
                    ga_entries=self.history[history_start_idx:],
                )
                return dict(best_current["hp"]), best_current["fitness"]

            self.best_round_fitness.append(0.0)
            telemetry.append_ga_candidates_rows(
                client_id=client_id,
                client_round=self.round_counter,
                ga_entries=self.history[history_start_idx:],
            )
            fallback = candidates_pool[0] if candidates_pool else self._create_individual()
            return dict(fallback), 0.0

        # Default GA 
        else:
            self.population[:] = candidates_pool[: config.POPULATION_SIZE]

            for ind in self.population:
                result = self._evaluate_rung(
                    ind,
                    global_state_dict,
                    epochs=config.EPOCHS,
                    subset_ratio=1.0,           # Full dataset
                    mu=config.RUNG1_MU,
                    rung_tag="full",
                )
                if result is None:
                    continue
                fitness, drift, val_acc = result
                ind.fitness.values = (fitness,)

                entry = self._history_entry(dict(ind), fitness, drift, val_acc, rung="full")
                self.history.append(entry)

            evaluated = [ind for ind in self.population if ind.fitness.valid]
            if not evaluated:
                # All candidates OOM'd on this visit; fall back to a fresh random HP.
                self.best_round_fitness.append(0.0)
                telemetry.append_ga_candidates_rows(
                    client_id=client_id,
                    client_round=self.round_counter,
                    ga_entries=self.history[history_start_idx:],
                )
                fallback = self._create_individual()
                return dict(fallback), 0.0

            sorted_pop = sorted(evaluated, key=lambda ind: ind.fitness.values[0], reverse=True)

            self.elite = []
            for ind in sorted_pop[:3]:
                self.elite.append({"hp": dict(ind), "fitness": ind.fitness.values[0]})

            self.best_round_fitness.append(float(sorted_pop[0].fitness.values[0]))

            telemetry.append_ga_candidates_rows(
                client_id=client_id,
                client_round=self.round_counter,
                ga_entries=self.history[history_start_idx:],
            )
            return dict(sorted_pop[0]), sorted_pop[0].fitness.values[0]
