import logging
import random
from datetime import datetime
from typing import Dict, Iterable

import numpy as np
import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from logging import INFO

from federatedgeneticalgorithm.task import build_model, trainset, testset, partition_class_distribution
from federatedgeneticalgorithm.config import config
from federatedgeneticalgorithm import telemetry
from federatedgeneticalgorithm.genetic_algorithm import HYPERPARAMS
from federatedgeneticalgorithm.federated_genetic_algorithm import FederatedGA
from federatedgeneticalgorithm.federated_baselines import FederatedRandomSearch, FederatedTPE

app = ServerApp()


def _metric_record_to_dict(metrics: MetricRecord | None) -> Dict[str, float]:
    if metrics is None:
        return {}
    return {str(k): float(v) if isinstance(v, (int, float)) else v for k, v in dict(metrics).items()}


class TelemetryFedAvg(FedAvg):
    """FedAvg with per-round metrics dumped to CSV via `telemetry`."""

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies_list = list(replies)
        arrays, metrics = super().aggregate_train(server_round, replies_list)
        telemetry.append_server_aggregated_row(
            server_round=server_round,
            phase="train",
            num_replies=len(replies_list),
            metrics=_metric_record_to_dict(metrics),
        )
        return arrays, metrics

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        replies_list = list(replies)
        metrics = super().aggregate_evaluate(server_round, replies_list)
        telemetry.append_server_aggregated_row(
            server_round=server_round,
            phase="evaluate",
            num_replies=len(replies_list),
            metrics=_metric_record_to_dict(metrics),
        )
        return metrics


class FederatedGAFedAvg(TelemetryFedAvg):
    """FedAvg variant that picks one HP per round via a server-side searcher
    and broadcasts it to all clients through the per-round ConfigRecord.

    Fitness is the aggregated eval-acc of the round. With use_delta_fitness it
    becomes the per-round improvement (eval_acc_after - eval_acc_before)
    instead. Neither signal is unbiased: absolute fitness under-scores HPs
    evaluated on a cold model, and delta over-credits whatever runs right
    after a crash or on the steep early curve. Both are kept for the ablation.
    """

    def __init__(self, searcher, *args, use_delta_fitness: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # `searcher` is duck-typed: any object with select_for_round(round) -> hp_dict,
        # record_fitness(fitness) -> info_dict, and attributes
        # {generation, current_idx, best_hp, best_fitness, population}.
        # Concrete impls: FederatedGA, FederatedRandomSearch, FederatedTPE.
        self.fed_ga = searcher
        self._tag = f"HPSearch:{type(searcher).__name__}"
        self._current_hp: Dict = {}
        self._prev_eval_acc: float = 0.0
        self._use_delta_fitness: bool = use_delta_fitness

    def configure_train(self, server_round, arrays, config_record, grid):
        hp = self.fed_ga.select_for_round(server_round)
        self._current_hp = hp
        # Inject HP into the ConfigRecord that ships with each train message
        config_record["fed_ga_hp_batch_size"] = int(hp["batch_size"])
        config_record["fed_ga_hp_optimizer"] = str(hp["optimizer"])
        config_record["fed_ga_hp_lr"] = float(hp["lr"])
        config_record["fed_ga_hp_weight_decay"] = float(hp["weight_decay"])
        config_record["fed_ga_hp_momentum"] = float(hp["momentum"])
        config_record["fed_ga_generation"] = int(self.fed_ga.generation)
        log(
            INFO,
            f"[{self._tag}] Round {server_round} -- broadcasting HP: "
            f"batch={hp['batch_size']}, opt={hp['optimizer']}, lr={hp['lr']}, "
            f"wd={hp['weight_decay']}, mom={hp['momentum']} "
            f"(gen={self.fed_ga.generation}, idx={self.fed_ga.current_idx})",
        )
        return super().configure_train(server_round, arrays, config_record, grid)

    def aggregate_evaluate(self, server_round, replies):
        replies_list = list(replies)
        metrics = super().aggregate_evaluate(server_round, replies_list)
        m_dict = _metric_record_to_dict(metrics)
        eval_acc = m_dict.get("eval-acc")
        if eval_acc is not None and self._current_hp:
            eval_acc_f = float(eval_acc)
            if self._use_delta_fitness:
                fitness = eval_acc_f - self._prev_eval_acc
            else:
                fitness = eval_acc_f
            info = self.fed_ga.record_fitness(fitness)
            log(
                INFO,
                f"[{self._tag}] Round {server_round} -- eval-acc={eval_acc_f:.4f}, "
                f"fitness(Δ)={fitness:+.4f} "
                f"(gen={info['generation']}, best_Δ_so_far={self.fed_ga.best_fitness:+.4f})",
            )
            self._prev_eval_acc = eval_acc_f
            if "evolved_to_generation" in info:
                log(
                    INFO,
                    f"[{self._tag}] >>> Evolved to generation {info['evolved_to_generation']}. "
                    f"Best HP (by Δ): {self.fed_ga.best_hp} Δ={self.fed_ga.best_fitness:+.4f}",
                )
        return metrics


def setup_file_logging(log_file: str = "training.log") -> logging.Handler:
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(handler)
    return handler


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run FedAvg for N rounds and save the final state_dict."""
    file_handler = setup_file_logging("training.log")
    start_time = datetime.now()

    # Server-side seeding so the global CNN init is deterministic across runs
    # with the same config.SEED. Client-side randomness (augmentation, GA, Ray
    # worker scheduling) is not covered here -- treat residual run-to-run
    # variance as the σ_implementação when comparing scenarios.
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    telemetry.initialize_run(force_new=True, record_config=True)

    fraction_train: float = config.FRACTION_TRAIN
    num_rounds: int = config.NUM_SERVER_ROUNDS
    num_partitions = int(context.run_config.get("num-supernodes", 10))

    train_hist = partition_class_distribution(trainset, num_partitions, seed=config.SEED, force_iid=False)
    test_hist = partition_class_distribution(testset, num_partitions, seed=config.SEED, force_iid=True)
    telemetry.save_partition_distribution(train_hist, test_hist, config.PARTITION_MODE, config.DIRICHLET_ALPHA)

    global_model = build_model()
    arrays = ArrayRecord(global_model.state_dict())

    # Mutually exclusive server-side HP search strategies: exactly one or zero may
    # be enabled. Multiple-enabled is a config error.
    server_side_flags = [
        ("ENABLE_FED_GA", config.ENABLE_FED_GA),
        ("ENABLE_FED_RANDOM_SEARCH", getattr(config, "ENABLE_FED_RANDOM_SEARCH", False)),
        ("ENABLE_FED_TPE", getattr(config, "ENABLE_FED_TPE", False)),
    ]
    enabled = [name for name, on in server_side_flags if on]
    if len(enabled) > 1:
        raise RuntimeError(
            f"At most one of ENABLE_FED_GA / ENABLE_FED_RANDOM_SEARCH / ENABLE_FED_TPE may be "
            f"True; got: {enabled}"
        )

    # Baseline HP that seeds the initial population (used by GA + Random Search;
    # TPE ignores seed_individuals, see federated_baselines.py docstring).
    baseline_hp = {
        "batch_size": int(config.DEFAULT_BATCH_SIZE),
        "optimizer": str(config.DEFAULT_OPTIMIZER),
        "lr": float(config.DEFAULT_LR),
        "weight_decay": float(config.DEFAULT_WEIGHT_DECAY),
        "momentum": float(config.DEFAULT_MOMENTUM),
    }
    searcher = None
    seed_baseline = bool(getattr(config, "FED_GA_SEED_BASELINE", True))
    use_delta_fitness = bool(getattr(config, "FED_GA_USE_DELTA_FITNESS", False))
    ga_seed_individuals = [baseline_hp] if seed_baseline else None
    if config.ENABLE_FED_GA:
        searcher = FederatedGA(
            hyperparams=HYPERPARAMS,
            pop_size=config.FED_GA_POPULATION_SIZE,
            mutation_prob=config.MUTATION_PROB,
            crossover_prob=config.CROSSOVER_PROB,
            tournament_size=config.TOURNAMENT_SIZE,
            seed=config.SEED,
            seed_individuals=ga_seed_individuals,
        )
        seed_tag = "baseline-seeded" if seed_baseline else "random-pop"
        log(INFO, f"[FedGA] Initial population ({seed_tag}): {searcher.population}")
    elif getattr(config, "ENABLE_FED_RANDOM_SEARCH", False):
        searcher = FederatedRandomSearch(
            hyperparams=HYPERPARAMS,
            pop_size=config.FED_GA_POPULATION_SIZE,
            seed=config.SEED,
            seed_individuals=ga_seed_individuals,
        )
        seed_tag = "baseline-seeded" if seed_baseline else "random-pop"
        log(INFO, f"[FedRandomSearch] Initial snapshot ({seed_tag}): {searcher.population}")
    elif getattr(config, "ENABLE_FED_TPE", False):
        searcher = FederatedTPE(
            hyperparams=HYPERPARAMS,
            pop_size=config.FED_GA_POPULATION_SIZE,
            seed=config.SEED,
            # TPE ignores seed_individuals by design; pass None for clarity.
            seed_individuals=None,
        )
        log(INFO, f"[FedTPE] Initial snapshot (random): {searcher.population}")

    if searcher is not None:
        strategy = FederatedGAFedAvg(
            searcher, fraction_train=fraction_train, use_delta_fitness=use_delta_fitness
        )
        if use_delta_fitness:
            log(INFO, "[HPSearch] Fitness signal: Δeval-acc (delta vs previous round)")
    else:
        strategy = TelemetryFedAvg(fraction_train=fraction_train)

    log(INFO, "=" * 60)
    log(INFO, f"Starting Federated Learning - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(INFO, "GPU activated" if torch.cuda.is_available() else "GPU unavailable")
    log(INFO, "=" * 60)
    log(INFO, f"Configuration: fraction_train={fraction_train}, num_rounds={num_rounds}")
    log(INFO, f"Partitioning: mode={config.PARTITION_MODE}, alpha={config.DIRICHLET_ALPHA}")
    log(INFO, f"ENABLE_GA={config.ENABLE_GA}, ENABLE_FED_GA={config.ENABLE_FED_GA}, mu={config.LOCAL_TRAIN_MU}, lambda={config.FITNESS_DRIFT_PENALTY_LAMBDA}")

    result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=num_rounds)

    duration = (datetime.now() - start_time).total_seconds()
    log(INFO, "=" * 60)
    log(INFO, "Federated Learning finished. Saving final model to disk...")
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    log(INFO, "Model saved as 'final_model.pt'.")
    log(INFO, f"Total execution time: {duration:.2f}s ({duration / 60:.2f} min)")
    log(INFO, "=" * 60 + "\n")

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()
