import torch
import logging
from datetime import datetime

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flwr.common.logger import log
from logging import INFO

from federatedgeneticalgorithm.task import CNN

app = ServerApp()


def setup_file_logging(log_file: str = "training.log") -> logging.Handler:
    """Attach a FileHandler to the root logger (append mode)."""
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    logging.getLogger().addHandler(file_handler)
    return file_handler


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Initialize global state, run FedAvg for N rounds, and save the final state_dict to disk."""
    file_handler = setup_file_logging("training.log")
    start_time = datetime.now()

    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]

    global_model = CNN()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(fraction_train=fraction_train)

    log(INFO, "=" * 60)
    log(INFO, f"Starting Federated Learning - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(INFO, "=" * 60)
    log(INFO, f"Configuration: fraction_train={fraction_train}, num_rounds={num_rounds}")

    result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=num_rounds)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    log(INFO, "=" * 60)
    log(INFO, "Federated Learning finished. Saving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    final_model = "final_model.pt"
    torch.save(state_dict, final_model)
    log(INFO, f"Model saved as '{final_model}'.")
    log(INFO, f"Total execution time: {duration:.2f}s ({duration/60:.2f} min)")
    log(INFO, "=" * 60 + "\n")

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()