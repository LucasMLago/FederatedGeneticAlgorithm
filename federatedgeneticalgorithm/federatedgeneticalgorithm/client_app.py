import torch
from typing import Dict
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federatedgeneticalgorithm.task import CNN, build_dataloaders, trainset, testset
from federatedgeneticalgorithm.task import test as test_fn
from federatedgeneticalgorithm.task import train as train_fn
from federatedgeneticalgorithm.task import get_partition
from federatedgeneticalgorithm.genetic_algorithm import GeneticAlgorithm
from federatedgeneticalgorithm.config import config

from logging import INFO
from flwr.common.logger import log

app = ClientApp()

# Persist GA instance per partition to maintain population history across rounds
CLIENT_GA_instances: Dict[int, GeneticAlgorithm] = {}


@app.train()
def train(msg: Message, context: Context):
    """Flow: load global weights -> run 1 GA update -> train locally -> return weights + metrics."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    model = CNN()
    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_state_dict)

    device = torch.device(config.DEVICE)

    local_trainset = get_partition(trainset, partition_id, num_partitions, seed=config.SEED)
    local_testset = get_partition(testset, partition_id, num_partitions, seed=config.SEED)

    best_hp = {}

    if config.ENABLE_GA:
        log(INFO, f"[Client {partition_id}] Starting GA Round...")

        if partition_id not in CLIENT_GA_instances:
            log(INFO, f"[Client {partition_id}] Initializing new Genetic Algorithm instance.")
            CLIENT_GA_instances[partition_id] = GeneticAlgorithm(model, local_trainset, local_testset)

        # Run incremental GA to refine hyperparameters before local training
        ga = CLIENT_GA_instances[partition_id]
        best_hp = ga.run_round_updates(global_state_dict=global_state_dict)
    else:
        log(INFO, f"[Client {partition_id}] GA Disabled. Using fixed default hyperparameters.")
        best_hp = {
            "batch_size": config.DEFAULT_BATCH_SIZE,
            "optimizer": config.DEFAULT_OPTIMIZER,
            "lr": config.DEFAULT_LR,
            "weight_decay": config.DEFAULT_WEIGHT_DECAY,
            "momentum": config.DEFAULT_MOMENTUM,
        }

    batch_size = best_hp["batch_size"]
    optimizer = best_hp["optimizer"]
    lr = best_hp["lr"]
    weight_decay = best_hp["weight_decay"]
    momentum = best_hp.get("momentum", 0.0)

    if momentum == 0.0:
        log(
            INFO,
            f"[Client {partition_id}] Best HP selected: batch={batch_size}, lr={lr}, optimizer={optimizer}, weight_decay={weight_decay}",
        )
    else:
        log(
            INFO,
            f"[Client {partition_id}] Best HP selected: batch={batch_size}, lr={lr}, optimizer={optimizer}, weight_decay={weight_decay}, momentum={momentum}",
        )

    model.load_state_dict(global_state_dict)
    model.to(device)

    trainloader, valloader, _ = build_dataloaders(
        local_trainset, local_testset, batch_size=batch_size, seed=config.SEED
    )

    local_epochs = context.run_config.get("local-epochs", 1)

    # Final local training using the optimized hyperparameters
    train_metrics = train_fn(
        model,
        trainloader,
        local_epochs,
        lr,
        device,
        optimizer,
        weight_decay,
        momentum,
        mu=0.0,
        global_state_dict=global_state_dict,
    )

    val_loss, val_acc = test_fn(model, valloader, device)

    log(
        INFO,
        f"[Client {partition_id}] Training completed: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}",
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train-loss": train_metrics["loss"],
        "train-accuracy": train_metrics["accuracy"],
        "val-loss": val_loss,
        "val-accuracy": val_acc,
        "num-examples": int(len(trainloader.dataset)),
        "hp_lr": float(lr),
        "hp_batch_size": int(batch_size),
        "hp_weight_decays": float(weight_decay),
        "hp_momentum": float(momentum),
        # "hp_optmizer": str(optimizer)
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on the local test partition and return aggregatable metrics."""
    model = CNN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device(config.DEVICE)
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    local_trainset = get_partition(trainset, partition_id, num_partitions, seed=config.SEED)
    local_testset = get_partition(testset, partition_id, num_partitions, seed=config.SEED)

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
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return Message(content=content, reply_to=msg)
