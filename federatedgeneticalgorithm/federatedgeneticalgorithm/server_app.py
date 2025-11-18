"""FederatedGeneticAlgorithm: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flwr.common.logger import log
from logging import INFO, WARNING

from federatedgeneticalgorithm.task import Net
from federatedgeneticalgorithm.strategy import FedAvgProbabilistic

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    client_probabilities_str: str = context.run_config.get(
        "client-selection-probabilities",
        None
    )

    # Parse client probabilities from string
    client_probabilities = []
    for prob in client_probabilities_str.split(","):
        client_probabilities.append(float(prob))

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize strategy with probabilistic client selection
    log(INFO, f"Using probabilistic client selection")
    log(INFO, f"Client probabilities: {client_probabilities}")
    strategy = FedAvgProbabilistic(
        client_probabilities=client_probabilities,
        fraction_train=fraction_train
    )

    log(INFO, "\n\n")

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        # train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
