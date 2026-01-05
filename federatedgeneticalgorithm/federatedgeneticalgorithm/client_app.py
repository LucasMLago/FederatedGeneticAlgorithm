"""FederatedGeneticAlgorithm: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federatedgeneticalgorithm.task import CNN, build_dataloaders, trainset, testset
from federatedgeneticalgorithm.task import test as test_fn
from federatedgeneticalgorithm.task import train as train_fn
from federatedgeneticalgorithm.task import get_partition
from federatedgeneticalgorithm.genetic_algorithm import GeneticAlgorithm

import sys
sys.path.append("../")

from config import config

# Flower ClientApp
app = ClientApp()

CLIENT_STATE: dict = {"batch_size": None}

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = CNN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Get IID partitions of train and test datasets
    local_trainset = get_partition(trainset, partition_id, num_partitions, seed=config.SEED)
    local_testset = get_partition(testset, partition_id, num_partitions, seed=config.SEED)

    # Run genetic algorithm to find best hyperparameters
    print(f"[Client {partition_id}] Local training set size: {len(local_trainset)}")
    print(f"[Client {partition_id}] Local test set size: {len(local_testset)}")
    print(f"[Client {partition_id}] Running genetic algorithm to optimize hyperparameters...")
    
    ga = GeneticAlgorithm(model, local_trainset, local_testset)
    pop, log = ga.run()
    best_individual = ga.get_best_individuals(pop, k=1)[0]
    print(f"[Client {partition_id}] Genetic algorithm completed.")
    print(f"[Client {partition_id}] best individual fitness: {log.select('max')[-1]:.4f}")

    
    print(f"[Client {partition_id}] Best hyperparameters found:")
    print(f"  - batch_size: {best_individual['batch_size']}")
    print(f"  - optimizer: {best_individual['optimizer']}")
    print(f"  - lr: {best_individual['lr']}")
    print(f"  - weight_decay: {best_individual['weight_decay']}")
    if 'momentum' in best_individual:
        print(f"  - momentum: {best_individual['momentum']}")

    # Extract best hyperparameters
    batch_size = best_individual["batch_size"]
    optimizer = best_individual["optimizer"]
    lr = best_individual["lr"]
    weight_decay = best_individual["weight_decay"]
    momentum = best_individual.get("momentum", None)

    # Reload initial weights (GA already trained, we want to train from server weights)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)

    # Build dataloaders with optimized batch size
    trainloader, valloader, _ = build_dataloaders(
        local_trainset, 
        local_testset, 
        batch_size=batch_size, 
        seed=config.SEED
    )

    # Update client state
    CLIENT_STATE.update({
        "batch_size": batch_size,
        "optimizer": optimizer,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "partition_id": partition_id,
        "num_partitions": num_partitions,
    })

    # Train with optimized hyperparameters
    local_epochs = context.run_config["local-epochs"]
    train_metrics = train_fn(
        model,
        trainloader,
        local_epochs,
        lr,
        device,
        optimizer,
        weight_decay,
        momentum,
    )
    
    # Validate
    val_loss, val_acc = test_fn(model, valloader, device)

    print(f"[Client {partition_id}] Training completed:")
    print(f"  - Train loss: {train_metrics['loss']:.4f}, Train acc: {train_metrics['accuracy']:.4f}")
    print(f"  - Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train-loss": train_metrics["loss"],
        "train-accuracy": train_metrics["accuracy"],
        "val-loss": val_loss,
        "val-accuracy": val_acc,
        "num-examples": int(len(trainloader.dataset)),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = CNN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Get IID partitions
    local_trainset = get_partition(trainset, partition_id, num_partitions, seed=config.SEED)
    local_testset = get_partition(testset, partition_id, num_partitions, seed=config.SEED)

    batch_size = CLIENT_STATE.get("batch_size") or 32
    _, _, testloader = build_dataloaders(local_trainset, local_testset, batch_size=batch_size, seed=config.SEED)

    eval_loss, eval_acc = test_fn(
        model,
        testloader,
        device,
    )

    metrics = {
        "eval-loss": eval_loss,
        "eval-acc": eval_acc,
        "num-examples": int(len(testloader.dataset)),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
