import logging
import math
import random
from datetime import datetime
from typing import Dict, Iterable

import numpy as np
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.common import MessageType
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import sample_nodes
from logging import INFO

from federatedgeneticalgorithm.task import build_model, trainset, testset, partition_class_distribution
from federatedgeneticalgorithm.config import config
from federatedgeneticalgorithm import telemetry
from federatedgeneticalgorithm.genetic_algorithm import HYPERPARAMS
from federatedgeneticalgorithm.federated_genetic_algorithm import FederatedGA
from federatedgeneticalgorithm.federated_baselines import FederatedRandomSearch, FederatedTPE, FedEx
from federatedgeneticalgorithm.federated_fedpop import FedPop, incumbent

app = ServerApp()


def _metric_record_to_dict(metrics: MetricRecord | None) -> Dict[str, float]:
    if metrics is None:
        return {}
    return {str(k): float(v) if isinstance(v, (int, float)) else v for k, v in dict(metrics).items()}


class TelemetryFedAvg(FedAvg):
    """FedAvg that also writes the aggregated metrics to CSV."""

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies_list = list(replies)
        arrays, metrics = super().aggregate_train(server_round, replies_list)
        self._after_aggregate_train(server_round, replies_list)
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
        row = _metric_record_to_dict(metrics)
        row.update(self._extra_evaluate_row(replies_list))
        telemetry.append_server_aggregated_row(
            server_round=server_round,
            phase="evaluate",
            num_replies=len(replies_list),
            metrics=row,
        )
        return metrics

    def _extra_evaluate_row(self, replies_list) -> Dict[str, float]:
        return {}

    def _after_aggregate_train(self, server_round: int, replies_list) -> None:
        pass


def pooled_fitness_val(replies_list) -> Dict[str, float]:
    """Val acc/loss pooled over clients by split size. Raises rather than fall back to the test split."""
    total = 0
    acc_sum = 0.0
    loss_sum = 0.0
    for reply in replies_list:
        if reply.has_error():
            continue
        m = reply.content["metrics"]
        if "fitness-val-acc" not in m:
            raise RuntimeError(
                "FED_FITNESS_SPLIT='val' but an evaluate reply has no fitness-val-acc; "
                "refusing to score the broadcast HP on the test partition."
            )
        n = int(m["fitness-val-num-examples"])
        total += n
        acc_sum += float(m["fitness-val-acc"]) * n
        loss_sum += float(m["fitness-val-loss"]) * n
    if total == 0:
        raise RuntimeError("No successful evaluate replies to compute the fitness split from.")
    return {"fitness-val-acc": acc_sum / total, "fitness-val-loss": loss_sum / total}


class FederatedGAFedAvg(TelemetryFedAvg):
    """Server-side searcher picks one HP per round and every client trains with it."""

    def __init__(self, searcher, *args, use_delta_fitness: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # "val" = clients' held-out split; "test" only to reproduce the old runs
        self._fitness_split: str = str(config.FED_FITNESS_SPLIT)
        if self._fitness_split not in ("val", "test"):
            raise ValueError(f"FED_FITNESS_SPLIT must be 'val' or 'test', got {self._fitness_split!r}")
        self._pooled_val: Dict[str, float] = {}
        # FederatedGA, FederatedRandomSearch or FederatedTPE, despite the attribute name
        self.fed_ga = searcher
        self._tag = f"HPSearch:{type(searcher).__name__}"
        self._current_hp: Dict = {}
        self._prev_signal: float = 0.0
        self._use_delta_fitness: bool = use_delta_fitness

    def configure_train(self, server_round, arrays, config_record, grid):
        hp = self.fed_ga.select_for_round(server_round)
        self._current_hp = hp
        # read on the client side by _extract_fed_ga_hp
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

    def _extra_evaluate_row(self, replies_list) -> Dict[str, float]:
        return self._pooled_val

    def aggregate_evaluate(self, server_round, replies):
        replies_list = list(replies)
        # before super() so it ends up in the telemetry row
        self._pooled_val = pooled_fitness_val(replies_list) if self._fitness_split == "val" else {}
        metrics = super().aggregate_evaluate(server_round, replies_list)
        m_dict = _metric_record_to_dict(metrics)
        eval_acc = m_dict.get("eval-acc")
        if self._fitness_split == "val":
            signal = self._pooled_val["fitness-val-acc"]
        else:
            signal = None if eval_acc is None else float(eval_acc)
        if signal is not None and self._current_hp:
            if self._use_delta_fitness:
                fitness = signal - self._prev_signal
            else:
                fitness = signal
            info = self.fed_ga.record_fitness(fitness)
            log(
                INFO,
                f"[{self._tag}] Round {server_round} -- eval-acc={float(eval_acc or 0.0):.4f}, "
                f"fitness[{self._fitness_split}]={signal:.4f}, "
                f"fitness(Δ)={fitness:+.4f} "
                f"(gen={info['generation']}, best_Δ_so_far={self.fed_ga.best_fitness:+.4f})",
            )
            self._prev_signal = signal
            if "evolved_to_generation" in info:
                log(
                    INFO,
                    f"[{self._tag}] >>> Evolved to generation {info['evolved_to_generation']}. "
                    f"Best HP (by Δ): {self.fed_ga.best_hp} Δ={self.fed_ga.best_fitness:+.4f}",
                )
        return metrics


FEDEX_ROUND_HEADERS = [
    "server_round", "fedex-refine-error", "fedex-baseline", "fedex-entropy", "fedex-mle-prob",
    "mle_batch_size", "mle_optimizer", "mle_lr", "mle_weight_decay", "mle_momentum",
]


class FedExFedAvg(TelemetryFedAvg):
    """FedEx: every sampled client trains with its own HP, drawn from the server's distribution."""

    def __init__(self, fedex: FedEx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fedex = fedex
        self._assigned: Dict[int, tuple] = {}  # node id -> grid indices of the HP it was sent

    def configure_train(self, server_round, arrays, config_record, grid):
        messages = list(super().configure_train(server_round, arrays, config_record, grid))
        self._assigned = {}
        out = []
        for msg in messages:
            node_id = msg.metadata.dst_node_id
            idx, hp = self.fedex.sample()
            self._assigned[node_id] = idx
            cfg = ConfigRecord(dict(config_record))
            # same keys as the broadcast searchers, read by _extract_fed_ga_hp
            cfg["fed_ga_hp_batch_size"] = int(hp["batch_size"])
            cfg["fed_ga_hp_optimizer"] = str(hp["optimizer"])
            cfg["fed_ga_hp_lr"] = float(hp["lr"])
            cfg["fed_ga_hp_weight_decay"] = float(hp["weight_decay"])
            cfg["fed_ga_hp_momentum"] = float(hp["momentum"])
            record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: cfg})
            out.append(Message(content=record, message_type=MessageType.TRAIN, dst_node_id=node_id))
            log(
                INFO,
                f"[FedEx] Round {server_round} -- node {node_id}: batch={hp['batch_size']}, "
                f"opt={hp['optimizer']}, lr={hp['lr']}, wd={hp['weight_decay']}, mom={hp['momentum']}",
            )
        return out

    def _after_aggregate_train(self, server_round: int, replies_list) -> None:
        assigned, errors, weights = [], [], []
        for reply in replies_list:
            if reply.has_error():
                continue
            m = reply.content["metrics"]
            if "fedex-val-acc" not in m:
                raise RuntimeError("FedEx needs fedex-val-acc in every train reply (ENABLE_FEDEX off on the client?)")
            assigned.append(self._assigned[reply.metadata.src_node_id])
            errors.append(1.0 - float(m["fedex-val-acc"]))
            weights.append(float(m["fedex-val-num-examples"]))
        if not assigned:
            return
        info = self.fedex.step(assigned, errors, weights)
        mle = self.fedex.mle()
        log(
            INFO,
            f"[FedEx] Round {server_round} -- refine error={info['fedex-refine-error']:.4f}, "
            f"baseline={info['fedex-baseline']:.4f}, entropy={info['fedex-entropy']:.3f}, "
            f"MLE HP (p={info['fedex-mle-prob']:.3f}): {mle}",
        )
        if config.ENABLE_TELEMETRY_EXPORT:
            row = {"server_round": server_round, **info, **{f"mle_{k}": v for k, v in mle.items()}}
            telemetry._append_csv_row(telemetry.get_run_dir() / "fedex_rounds.csv", FEDEX_ROUND_HEADERS, row)


FEDPOP_ROUND_HEADERS = ["server_round", "incumbent", "fedpop_g", *[f"val_loss_{i}" for i in range(16)]]
FEDPOP_CLIENT_HEADERS = ["server_round", "process", "local", "partition", "batch_size", "optimizer", "lr",
                         "weight_decay", "momentum", "val_loss", "val_acc"]


def fedpop_losses(node_ids, replies):
    """Held-out loss of each client HP slot k, matched by the node that trained it (inf if it failed)."""
    losses, by_local = [float("inf")] * len(node_ids), {}
    for reply in replies:
        if reply.has_error():
            continue
        m = reply.content["metrics"]
        if "fedpop-val-loss" not in m:
            raise RuntimeError("FedPop needs fedpop-val-loss in every train reply (ENABLE_FEDPOP off on the client?)")
        k = node_ids.index(reply.metadata.src_node_id)
        losses[k] = float(m["fedpop-val-loss"])
        by_local[k] = m
    return losses, by_local


class FedPopFedAvg(TelemetryFedAvg):
    """FedPop: N_c FedAvg trainings run side by side on the same sampled clients each round.

    Every process trains on the round's sampled clients, each client with its own HP (FedPop-L inside
    the process); every T_g rounds FedPop-G copies weights from the best processes to the worst. The
    process with the lowest pooled validation loss in a round is the one evaluated and logged, so the
    per-round telemetry has the same shape as for the other designs."""

    def __init__(self, fedpop: FedPop, *args, global_every: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.fedpop = fedpop
        self.global_every = global_every

    def _train_messages(self, proc_idx, node_ids, server_round, config_record):
        proc = self.fedpop.processes[proc_idx]
        out = []
        for k, node_id in enumerate(node_ids):
            hp = proc.locals[k]
            cfg = ConfigRecord(dict(config_record))
            cfg["server-round"] = server_round
            cfg["fed_ga_hp_batch_size"] = int(hp["batch_size"])
            cfg["fed_ga_hp_optimizer"] = str(hp["optimizer"])
            cfg["fed_ga_hp_lr"] = float(hp["lr"])
            cfg["fed_ga_hp_weight_decay"] = float(hp["weight_decay"])
            cfg["fed_ga_hp_momentum"] = float(hp["momentum"])
            record = RecordDict({self.arrayrecord_key: proc.arrays, self.configrecord_key: cfg})
            out.append(Message(content=record, message_type=MessageType.TRAIN, dst_node_id=node_id))
        return out

    def start(self, grid, initial_arrays, num_rounds=3, timeout=3600, train_config=None, evaluate_config=None,
              evaluate_fn=None):
        result = Result()
        for proc in self.fedpop.processes:
            proc.arrays = initial_arrays
        run_dir = telemetry.get_run_dir() if config.ENABLE_TELEMETRY_EXPORT else None
        for rnd in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", rnd, num_rounds)
            # same sampling calls as FedAvg (train, then evaluate), so seeds stay paired with the other designs
            num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
            node_ids, _ = sample_nodes(grid, self.min_available_nodes, max(num_nodes, self.min_train_nodes))
            round_losses, round_metrics = [], []
            for i, proc in enumerate(self.fedpop.processes):
                replies = list(grid.send_and_receive(self._train_messages(i, node_ids, rnd, ConfigRecord()),
                                                     timeout=timeout))
                losses, by_local = fedpop_losses(node_ids, replies)
                pooled = sum(losses[k] * int(m["fedpop-val-num-examples"]) for k, m in by_local.items())
                n_val = sum(int(m["fedpop-val-num-examples"]) for m in by_local.values())
                for k, m in by_local.items():
                    if run_dir is not None:
                        hp = proc.locals[k]
                        telemetry._append_csv_row(run_dir / "fedpop_clients.csv", FEDPOP_CLIENT_HEADERS, {
                            "server_round": rnd, "process": i, "local": k, "partition": int(m["fedpop-partition"]),
                            **hp, "val_loss": losses[k], "val_acc": float(m["fedpop-val-acc"])})
                arrays, metrics = FedAvg.aggregate_train(self, rnd, replies)
                if arrays is not None:
                    proc.arrays = arrays
                round_loss = pooled / n_val if n_val else None
                if round_loss is not None and math.isfinite(round_loss):
                    proc.history.append(round_loss)
                else:
                    round_loss = None
                round_losses.append(round_loss)
                round_metrics.append(metrics)
                self.fedpop.local_step(proc, losses, rnd, num_rounds)
            best = incumbent(round_losses)
            best_arrays = self.fedpop.processes[best].arrays
            log(INFO, f"[FedPop] Round {rnd} -- val loss per process: "
                      + ", ".join("-" if v is None else f"{v:.4f}" for v in round_losses) + f"; reported: {best}")
            telemetry.append_server_aggregated_row(server_round=rnd, phase="train", num_replies=len(node_ids),
                                                   metrics=_metric_record_to_dict(round_metrics[best]))
            pairs = []
            if rnd % self.global_every == 0 and rnd < num_rounds:
                pairs = self.fedpop.global_step(rnd, num_rounds)
                log(INFO, f"[FedPop] Round {rnd} -- FedPop-G (replaced, source): {pairs}")
            if run_dir is not None:
                row = {"server_round": rnd, "incumbent": best, "fedpop_g": str(pairs)}
                row.update({f"val_loss_{i}": v for i, v in enumerate(round_losses)})
                telemetry._append_csv_row(run_dir / "fedpop_rounds.csv", FEDPOP_ROUND_HEADERS, row)
            eval_replies = grid.send_and_receive(
                messages=self.configure_evaluate(rnd, best_arrays, ConfigRecord(), grid), timeout=timeout)
            self.aggregate_evaluate(rnd, eval_replies)
            result.arrays = best_arrays
        return result


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
    file_handler = setup_file_logging("training.log")
    start_time = datetime.now()

    # only makes the global model init reproducible; client-side randomness
    # (augmentation, GA, Ray scheduling) still varies between runs
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

    server_side_flags = [
        ("ENABLE_FED_GA", config.ENABLE_FED_GA),
        ("ENABLE_FED_RANDOM_SEARCH", getattr(config, "ENABLE_FED_RANDOM_SEARCH", False)),
        ("ENABLE_FED_TPE", getattr(config, "ENABLE_FED_TPE", False)),
        ("ENABLE_FEDEX", getattr(config, "ENABLE_FEDEX", False)),
        ("ENABLE_FEDPOP", getattr(config, "ENABLE_FEDPOP", False)),
    ]
    enabled = [name for name, on in server_side_flags if on]
    if len(enabled) > 1:
        raise RuntimeError(
            f"At most one of ENABLE_FED_GA / ENABLE_FED_RANDOM_SEARCH / ENABLE_FED_TPE / ENABLE_FEDEX / "
            f"ENABLE_FEDPOP may be "
            f"True; got: {enabled}"
        )

    # seeds the initial population of GA and random search (TPE ignores it)
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
            elitism=bool(config.FED_GA_ELITISM),
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
            seed_individuals=None,
        )
        log(INFO, f"[FedTPE] Initial snapshot (random): {searcher.population}")

    if getattr(config, "ENABLE_FEDPOP", False):
        num_train = max(int(num_partitions * fraction_train), 2)
        fedpop = FedPop(HYPERPARAMS, num_configs=int(config.FEDPOP_NUM_CONFIGS), num_clients=num_train,
                        seed=config.SEED)
        global_every = max(1, round(float(config.FEDPOP_GLOBAL_EVERY_FRACTION) * num_rounds))
        log(INFO, f"[FedPop] {config.FEDPOP_NUM_CONFIGS} processes, {num_train} client HPs each, "
                  f"FedPop-G every {global_every} rounds")
        strategy = FedPopFedAvg(fedpop, fraction_train=fraction_train, global_every=global_every)
    elif getattr(config, "ENABLE_FEDEX", False):
        fedex = FedEx(
            hyperparams=HYPERPARAMS,
            seed=config.SEED,
            sched=str(config.FEDEX_SCHED),
            baseline_discount=config.FEDEX_BASELINE_DISCOUNT,
        )
        log(INFO, f"[FedEx] sched={config.FEDEX_SCHED}, baseline discount={fedex.baseline_discount:.3f}")
        strategy = FedExFedAvg(fedex, fraction_train=fraction_train)
    elif searcher is not None:
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
