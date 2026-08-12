import gc
import time
from logging import INFO
from typing import Dict

import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.logger import log

from federatedgeneticalgorithm.task import build_model, build_dataloaders, trainset, testset
from federatedgeneticalgorithm.task import test as test_fn
from federatedgeneticalgorithm.task import train as train_fn
from federatedgeneticalgorithm.task import get_partition
from federatedgeneticalgorithm.genetic_algorithm import GeneticAlgorithm
from federatedgeneticalgorithm.config import config
from federatedgeneticalgorithm import telemetry

app = ClientApp()

# Per-partition GA instance; also recreated from disk inside train() because
# Ray's actor pool can recycle the worker between rounds.
CLIENT_GA_instances: Dict[int, GeneticAlgorithm] = {}
CLIENT_LOCAL_ROUND_COUNTER: Dict[int, int] = {}


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_local_datasets(partition_id: int, num_partitions: int):
    local_trainset = get_partition(trainset, partition_id, num_partitions, seed=config.SEED)
    # Test always IID so per-client eval is representative of the task.
    local_testset = get_partition(testset, partition_id, num_partitions, seed=config.SEED, force_iid=True)
    return local_trainset, local_testset


def _ga_state_path(partition_id: int) -> str:
    # Ray can spin up a fresh actor between rounds, so we persist GA state
    # (round counter, surrogate, history) on disk to survive recycles.
    return str(telemetry.get_run_dir() / "ga_state" / f"partition_{partition_id}.pkl")


def _extract_fed_ga_hp(msg: Message):
    """If the server attached a federated-GA HP to the ConfigRecord, return it."""
    try:
        cfg = msg.content["config"]
    except Exception:
        return None
    if "fed_ga_hp_lr" not in cfg:
        return None
    return {
        "batch_size": int(cfg["fed_ga_hp_batch_size"]),
        "optimizer": str(cfg["fed_ga_hp_optimizer"]),
        "lr": float(cfg["fed_ga_hp_lr"]),
        "weight_decay": float(cfg["fed_ga_hp_weight_decay"]),
        "momentum": float(cfg["fed_ga_hp_momentum"]),
    }


def _log_selected_hyperparams(
    partition_id: int,
    best_fitness: float,
    batch_size: int,
    lr: float,
    optimizer: str,
    weight_decay: float,
    momentum: float,
) -> None:
    base = (
        f"[Client {partition_id}] Best HP selected (Fitness: {best_fitness:.4f}): "
        f"batch={batch_size}, lr={lr}, optimizer={optimizer}, weight_decay={weight_decay}"
    )
    if momentum != 0.0:
        base += f", momentum={momentum}"
    log(INFO, base)


@app.train()
def train(msg: Message, context: Context):
    """Load global weights, optionally run one GA round, train locally, return new weights."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    model = build_model()
    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state_dict)

    device = _select_device()
    local_trainset, local_testset = _get_local_datasets(partition_id, num_partitions)

    best_hp: Dict = {}
    best_fitness = 0.0
    ga_time_s = 0.0
    ga_ran = False
    visit_start = time.perf_counter()

    # Server-broadcast GA path: the server attached the HP to the ConfigRecord; use it.
    fed_ga_hp = _extract_fed_ga_hp(msg)
    if fed_ga_hp is not None:
        best_hp = fed_ga_hp
        best_fitness = 0.0
        previous_round = CLIENT_LOCAL_ROUND_COUNTER.get(partition_id, 0)
        client_round = previous_round + 1
        CLIENT_LOCAL_ROUND_COUNTER[partition_id] = client_round
        log(
            INFO,
            f"[Client {partition_id}] FedGA-broadcast HP: "
            f"batch={best_hp['batch_size']}, opt={best_hp['optimizer']}, "
            f"lr={best_hp['lr']}, wd={best_hp['weight_decay']}, mom={best_hp['momentum']}",
        )
    elif config.ENABLE_GA:
        state_path = _ga_state_path(partition_id)
        if partition_id not in CLIENT_GA_instances:
            CLIENT_GA_instances[partition_id] = GeneticAlgorithm(model, local_trainset, local_testset)

        ga = CLIENT_GA_instances[partition_id]
        # Reload every call: if Ray served this partition from a different
        # actor last round, our in-memory state is stale.
        if ga.load_state(state_path):
            log(INFO, f"[Client {partition_id}] GA state restored from disk (round={ga.round_counter}, visits={ga.visit_counter}).")
        else:
            log(INFO, f"[Client {partition_id}] No saved GA state -- starting fresh.")

        ga.visit_counter += 1
        visit_counter = ga.visit_counter

        run_every = max(1, config.GA_RUN_EVERY_N_ROUNDS)
        should_run_ga = (
            visit_counter <= config.GA_WARMUP_ROUNDS
            or visit_counter % run_every == 0
            or ga.last_best_hp is None
        )

        if should_run_ga:
            log(INFO, f"[Client {partition_id}] Starting GA Round (visit {visit_counter})...")
            ga_start = time.perf_counter()
            best_hp, best_fitness = ga.run_round_updates(
                global_state_dict=global_state_dict, client_id=partition_id
            )
            ga_time_s = time.perf_counter() - ga_start
            ga.last_best_hp = dict(best_hp)
            ga.last_best_fitness = best_fitness
            ga_ran = True
            client_round = ga.round_counter
        else:
            log(INFO, f"[Client {partition_id}] Skipping GA this visit ({visit_counter}); reusing last best HP.")
            best_hp = dict(ga.last_best_hp)
            best_fitness = ga.last_best_fitness
            client_round = visit_counter
    else:
        log(INFO, f"[Client {partition_id}] GA disabled. Using fixed default hyperparameters.")
        best_hp = {
            "batch_size": config.DEFAULT_BATCH_SIZE,
            "optimizer": config.DEFAULT_OPTIMIZER,
            "lr": config.DEFAULT_LR,
            "weight_decay": config.DEFAULT_WEIGHT_DECAY,
            "momentum": config.DEFAULT_MOMENTUM,
        }
        best_fitness = 0.0
        previous_round = CLIENT_LOCAL_ROUND_COUNTER.get(partition_id, 0)
        client_round = previous_round + 1
        CLIENT_LOCAL_ROUND_COUNTER[partition_id] = client_round

    batch_size = best_hp["batch_size"]
    optimizer = best_hp["optimizer"]
    lr = best_hp["lr"]
    weight_decay = best_hp["weight_decay"]
    momentum = best_hp.get("momentum", 0.0)

    _log_selected_hyperparams(partition_id, best_fitness, batch_size, lr, optimizer, weight_decay, momentum)

    model.load_state_dict(global_state_dict)
    model.to(device)

    trainloader, valloader, testloader = build_dataloaders(
        local_trainset, local_testset, batch_size=batch_size, seed=config.SEED
    )
    local_epochs = config.LOCAL_EPOCHS

    train_start = time.perf_counter()
    train_metrics = train_fn(
        model,
        trainloader,
        local_epochs,
        lr,
        device,
        optimizer,
        weight_decay,
        momentum,
        mu=config.LOCAL_TRAIN_MU,
        global_state_dict=global_state_dict,
    )
    local_train_time_s = time.perf_counter() - train_start

    val_loss, val_acc = test_fn(model, valloader, device)
    local_test_loss, local_test_acc = test_fn(model, testloader, device)

    log(
        INFO,
        f"[Client {partition_id}] Training completed: "
        f"Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f} | "
        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} | "
        f"Test Loss={local_test_loss:.4f}, Test Acc={local_test_acc:.4f}",
    )

    total_visit_time_s = time.perf_counter() - visit_start
    num_examples = int(len(trainloader.dataset))
    model_record = ArrayRecord(model.state_dict())

    metrics = {
        "train-loss": train_metrics["loss"],
        "train-accuracy": train_metrics["accuracy"],
        "val-loss": val_loss,
        "val-accuracy": val_acc,
        "local-test-loss": local_test_loss,
        "local-test-accuracy": local_test_acc,
        "num-examples": num_examples,
        "ga-best-fitness": float(best_fitness),
    }
    telemetry_metrics = {
        **metrics,
        "ga_ran": int(ga_ran),
        "ga_time_s": round(ga_time_s, 3),
        "local_train_time_s": round(local_train_time_s, 3),
        "total_visit_time_s": round(total_visit_time_s, 3),
    }

    telemetry.append_client_round_row(
        client_id=partition_id,
        client_round=client_round,
        metrics=telemetry_metrics,
        best_hp=best_hp,
        best_fitness=best_fitness,
    )

    if config.ENABLE_GA and partition_id in CLIENT_GA_instances:
        try:
            CLIENT_GA_instances[partition_id].save_state(_ga_state_path(partition_id))
        except Exception as exc:
            log(INFO, f"[Client {partition_id}] Failed to save GA state: {exc!r}")

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    if torch.cuda.is_available():
        del model
        try:
            del trainloader, valloader, testloader
        except NameError:
            pass
        torch.cuda.empty_cache()
    gc.collect()

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the global model on the client's local test partition."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    model = build_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = _select_device()
    model.to(device)

    local_trainset, local_testset = _get_local_datasets(partition_id, num_partitions)
    eval_batch_size = 128
    _, _, testloader = build_dataloaders(local_trainset, local_testset, batch_size=eval_batch_size, seed=config.SEED)

    log(INFO, f"[Client {partition_id}] Evaluating model on local test set (Batch Size: {eval_batch_size})")
    eval_loss, eval_acc = test_fn(model, testloader, device)
    log(INFO, f"[Client {partition_id}] Evaluation result: Loss={eval_loss:.4f}, Acc={eval_acc:.4f}")

    metrics = {
        "eval-loss": eval_loss,
        "eval-acc": eval_acc,
        "num-examples": int(len(testloader.dataset)),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})

    if torch.cuda.is_available():
        del model
        try:
            del testloader
        except NameError:
            pass
        torch.cuda.empty_cache()
    gc.collect()

    return Message(content=content, reply_to=msg)
