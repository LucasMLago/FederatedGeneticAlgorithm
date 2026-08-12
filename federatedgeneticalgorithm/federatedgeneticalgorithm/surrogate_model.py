from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from federatedgeneticalgorithm.config import config


class SurrogateModel:
    """RandomForest regressor over (HP, drift) -> fitness (or post-agg proxy).

    The drift feature is what distinguishes HPs that look good locally from
    HPs that also produce aggregation-friendly updates.
    """

    def __init__(self, hyperparams_config: Dict[str, List]):
        self.model = RandomForestRegressor(n_estimators=10, random_state=config.SEED)
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder.fit(np.array(hyperparams_config["optimizers"]).reshape(-1, 1))
        self.ready = False
        # Pool-mean drift used when a caller can't supply one (e.g. Rung0 predictions).
        self.default_drift: float = 0.0
        # Which target the model was last fit on (mae_on_holdout compares against the same one).
        self._trained_target: str = "fitness"

    def _target_key(self) -> str:
        return str(getattr(config, "SURROGATE_TARGET", "fitness"))

    def _entry_target(self, entry: Dict, target_key: str) -> float:
        # Old pool entries predate drift/post_agg_proxy; fall back to "fitness"
        # so a reused shared_hp_pool.pkl doesn't blow up on cold start.
        value = entry.get(target_key)
        return float(value if value is not None else entry.get("fitness", 0.0))

    def _entry_drift(self, entry: Dict) -> float:
        return float(entry.get("drift", 0.0))

    def update(self, history: List[Dict]) -> None:
        """Retrain on pooled samples. No-op below 5 entries."""
        if len(history) < 5:
            return

        target_key = self._target_key()
        drifts = [self._entry_drift(e) for e in history]
        X = [self._hp_to_vector(e["hp"], drift=d) for e, d in zip(history, drifts)]
        y = [self._entry_target(e, target_key) for e in history]
        self.model.fit(X, y)
        self.ready = True
        self._trained_target = target_key
        if drifts:
            self.default_drift = float(np.mean(drifts))

    def predict_batch(
        self, hp_candidates: List[Dict], drift_estimate: Optional[float] = None
    ) -> List[float]:
        """Predict target for each candidate. Zeros if not trained yet."""
        if not self.ready:
            return [0.0] * len(hp_candidates)
        drift = self.default_drift if drift_estimate is None else float(drift_estimate)
        X = [self._hp_to_vector(hp, drift=drift) for hp in hp_candidates]
        return self.model.predict(X).tolist()

    def predict_with_uncertainty(
        self,
        hp_candidates: List[Dict],
        drift_estimate: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """(mean, std) per candidate; std comes from tree dispersion."""
        if not self.ready or not hp_candidates:
            return [(0.0, 0.0) for _ in hp_candidates]

        drift = self.default_drift if drift_estimate is None else float(drift_estimate)
        X = np.array([self._hp_to_vector(hp, drift=drift) for hp in hp_candidates])

        if not hasattr(self.model, "estimators_") or not self.model.estimators_:
            preds = self.model.predict(X).tolist()
            return [(float(p), 0.0) for p in preds]

        tree_preds = np.stack([est.predict(X) for est in self.model.estimators_], axis=0)
        return [
            (float(m), float(s))
            for m, s in zip(np.mean(tree_preds, axis=0), np.std(tree_preds, axis=0))
        ]

    def mae_on_holdout(self, pool: List[Dict], k: int = 10) -> Optional[float]:
        """MAE against the most recent `k` pool samples. None if untrained."""
        if not self.ready or not pool:
            return None
        k = max(1, min(int(k), len(pool)))
        sample = pool[-k:]
        target_key = self._trained_target
        preds: List[float] = []
        for entry in sample:
            preds.extend(self.predict_batch([entry["hp"]], drift_estimate=self._entry_drift(entry)))
        actuals = [self._entry_target(entry, target_key) for entry in sample]
        return float(np.mean(np.abs(np.array(preds) - np.array(actuals))))

    def _hp_to_vector(self, hp: Dict, drift: float = 0.0) -> List[float]:
        opt_vec = self.encoder.transform([[hp["optimizer"]]])[0].tolist()
        return [
            float(hp["batch_size"]),
            *opt_vec,
            float(hp["lr"]),
            float(hp["weight_decay"]),
            float(hp.get("momentum", 0.0)),
            float(drift),
        ]
