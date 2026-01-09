from typing import List, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

import numpy as np


class SurrogateModel:
    """Simple surrogate (RandomForest) predicting fitness from a fixed numeric Hyper Parameter vector."""

    def __init__(self, hyperparams_config: Dict[str, List]):
        self.model = None
        self.encoder = None
        self.ready = False

        if RandomForestRegressor:
            self.model = RandomForestRegressor(n_estimators=10, random_state=0)
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.encoder.fit(np.array(hyperparams_config["optimizers"]).reshape(-1, 1))

    def update(self, history: List[Dict]) -> None:
        """Retrain the surrogate using the collected history (requires a minimum sample count)."""
        # Ensure sufficient history before training the regressor
        if not self.model or len(history) < 5:
            return

        X = [self._hp_to_vector(entry["hp"]) for entry in history]
        y = [entry["fitness"] for entry in history]
        self.model.fit(X, y)
        self.ready = True

    def predict_batch(self, hp_candidates: List[Dict]) -> List[float]:
        """Predict fitness for candidates; returns zeros while the model is not ready."""
        if not self.ready:
            return [0.0] * len(hp_candidates)

        X = [self._hp_to_vector(hp) for hp in hp_candidates]
        return self.model.predict(X).tolist()

    def _hp_to_vector(self, hp: Dict) -> List[float]:
        """Encode an HP dict into a fixed numeric vector."""
        opt_val = 0
        if self.encoder:
            opt_val = int(self.encoder.transform([hp["optimizer"]])[0])

        return [
            float(hp["batch_size"]),
            float(opt_val),
            float(hp["lr"]),
            float(hp["weight_decay"]),
            float(hp.get("momentum", 0.0)),
        ]
