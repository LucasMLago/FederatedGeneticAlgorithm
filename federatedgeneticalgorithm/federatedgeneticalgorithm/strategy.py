"""Custom FedAvg strategy with probability-based client selection."""

from typing import List, Optional
import numpy as np
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp import Grid
from flwr.common.logger import log
from logging import INFO, WARNING


class FedAvgProbabilistic(FedAvg):
    """FedAvg strategy with probability-based client selection.
    
    Each client has a custom probability of being selected in each round.
    """

    def __init__(self, client_probabilities: List[float], fraction_train: float = 1.0, **kwargs):
        """Initialize the strategy.

        Parameters
        ----------
        client_probabilities : List[float]
            List of selection probabilities for each client. Must sum to 1.0.
        fraction_train : float
            Fraction of clients to select (will be used as number of clients to sample).
        """
        super().__init__(fraction_train=fraction_train, **kwargs)

        # Validate probabilities
        if not np.isclose(sum(client_probabilities), 1.0):
            raise ValueError(
                f"Client probabilities must sum to 1.0, got {sum(client_probabilities)}"
            )

        self.client_probabilities = np.array(client_probabilities)
        self.num_clients = len(client_probabilities)

        log(INFO, f"[Strategy] Initialized with {self.num_clients} clients")
        log(INFO, f"[Strategy] Client selection probabilities: {client_probabilities}")

    def sample_nodes(self, grid: Grid, num_nodes: Optional[int] = None) -> List[int]:
        """Sample client nodes based on their selection probabilities.

        Parameters
        ----------
        grid : Grid
            The grid containing all available nodes.
        num_nodes : Optional[int]
            Number of nodes to sample. If None, uses fraction_train.

        Returns
        -------
        List[int]
            List of selected node IDs.
        """
        available_nodes = grid.get_node_ids()

        # Calculate number of clients to select
        if num_nodes is None:
            num_nodes = max(1, int(len(available_nodes) * self.fraction_train))

        num_nodes = min(num_nodes, len(available_nodes))

        # Ensure we have probabilities for all available nodes
        if len(available_nodes) > len(self.client_probabilities):
            log(WARNING,
                f"[Strategy]: More nodes ({len(available_nodes)}) than "
                f"probabilities ({len(self.client_probabilities)}). "
                f"Using uniform probabilities for extra nodes."
            )
            extra_nodes = len(available_nodes) - len(self.client_probabilities)
            probabilities = np.concatenate([
                self.client_probabilities,
                np.ones(extra_nodes) / (len(self.client_probabilities) + extra_nodes)
            ])
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = self.client_probabilities[:len(available_nodes)]
            probabilities = probabilities / probabilities.sum()

        # Sample nodes according to probabilities
        selected_indices = np.random.choice(
            available_nodes,
            size=num_nodes,
            replace=False,
            p=probabilities
        )

        selected_nodes = [available_nodes[i] for i in selected_indices]

        log(INFO, f"[Strategy] Selected {len(selected_nodes)} clients: {selected_nodes}")
        log(INFO, f"[Strategy] Selection probabilities used: {probabilities[selected_indices]}")

        return selected_nodes
