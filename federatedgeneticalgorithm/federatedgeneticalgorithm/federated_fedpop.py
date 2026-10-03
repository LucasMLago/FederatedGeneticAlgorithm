"""FedPop (Chen et al., AAAI 2025) on the discrete HP grid: population bookkeeping only, no Flower.

N_c tuning processes train in parallel, each with its own model. Inside a process every active client
k uses its own HP vector beta_k, drawn near a base vector (FedPop-L replaces the worst ones each round
by perturbed copies of the best). Every T_g rounds the worst processes copy the weights and the
perturbed base of the best ones (FedPop-G). Follows github.com/HaokunChen245/FedPop
(tuners/pbt_wrs_mix.py, perturb_one_user): log-space perturbation of lr and weight decay by
U(-4eps, 4eps), momentum by U(-eps, eps), batch halved or doubled with probability 1/3 each, whole
vector resampled with probability p_re, quantile 0.3, discount 0.9. Continuous perturbations are
snapped to the nearest grid value; the optimizer only changes when the vector is resampled.
Uses its own RNG, so client sampling (global `random`) stays paired across designs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from federatedgeneticalgorithm.federated_genetic_algorithm import KEY_MAP

QUANTILE = 0.3
DISCOUNT = 0.9


def _snap(values: List[float], x: float, log: bool) -> float:
    if log:
        floor = min(v for v in values if v > 0) / 10  # weight decay 0 sits one decade below the grid
        key = [math.log10(v if v > 0 else floor) for v in values]
        x = math.log10(max(x, floor))
    else:
        key = list(values)
    return values[int(np.argmin([abs(k - x) for k in key]))]


def discounted_mean(trace: List[float], factor: float = DISCOUNT) -> float:
    weight = factor ** np.flip(np.arange(len(trace)), axis=0)
    return float(np.inner(trace, weight) / weight.sum())


def annealed(x0: float, rnd: int, num_rounds: int) -> float:
    """Cosine annealing of eps and p_re over the rounds (Eq. 4 of the paper)."""
    return x0 * 0.5 * (1.0 + math.cos(math.pi * rnd / num_rounds))


@dataclass
class Process:
    base: Dict
    locals: List[Dict]
    arrays: object = None  # model weights of this process (ArrayRecord on the server)
    history: List[float] = field(default_factory=list)  # weighted val loss per round since last FedPop-G


class FedPop:
    def __init__(self, hyperparams: Dict[str, List], num_configs: int, num_clients: int, seed: int = 0,
                 eps0: float = 0.1, resample0: float = 0.1) -> None:
        self.hyperparams = hyperparams
        self.grid = {KEY_MAP[k]: list(v) for k, v in hyperparams.items()}
        self.rng = np.random.default_rng(seed)
        self.eps0 = eps0
        self.resample0 = resample0
        self.num_clients = num_clients
        self.processes: List[Process] = []
        for _ in range(num_configs):
            base = self.sample()
            self.processes.append(Process(base=base, locals=self._locals_around(base, eps0)))

    def sample(self) -> Dict:
        return {k: v[int(self.rng.integers(len(v)))] for k, v in self.grid.items()}

    def perturb(self, hp: Dict, eps: float, resample_p: float) -> Dict:
        if self.rng.random() < resample_p:
            return self.sample()
        g, out = self.grid, dict(hp)
        out["lr"] = _snap(g["lr"], 10 ** (math.log10(hp["lr"]) + self.rng.uniform(-4 * eps, 4 * eps)), log=True)
        wd = hp["weight_decay"] if hp["weight_decay"] > 0 else min(v for v in g["weight_decay"] if v > 0) / 10
        out["weight_decay"] = _snap(g["weight_decay"], 10 ** (math.log10(wd) + self.rng.uniform(-4 * eps, 4 * eps)),
                                    log=True)
        out["momentum"] = _snap(g["momentum"], hp["momentum"] + self.rng.uniform(-eps, eps), log=False)
        p = self.rng.random()
        if p >= 2 / 3:
            out["batch_size"] = max(min(g["batch_size"]), hp["batch_size"] // 2)
        elif p > 1 / 3:
            out["batch_size"] = min(max(g["batch_size"]), hp["batch_size"] * 2)
        return out

    def _locals_around(self, base: Dict, eps: float) -> List[Dict]:
        return [self.perturb(base, eps, resample_p=0.0) for _ in range(self.num_clients)]

    def local_step(self, proc: Process, val_losses: List[float], rnd: int, num_rounds: int) -> None:
        """FedPop-L: the bottom quantile of a process's client HPs becomes perturbed copies of the top."""
        n = len(val_losses)
        order = sorted(range(n), key=lambda k: val_losses[k])
        q = math.ceil(QUANTILE * n)
        top, bottom = order[:q], order[-q:]
        eps, p_re = annealed(self.eps0, rnd, num_rounds), annealed(self.resample0, rnd, num_rounds)
        for kb in bottom:
            kt = top[int(self.rng.integers(len(top)))]
            proc.locals[kb] = self.perturb(proc.locals[kt], eps, p_re)

    def global_step(self, rnd: int, num_rounds: int) -> List[tuple[int, int]]:
        """FedPop-G: the bottom quantile of processes takes the weights and a perturbed base of the top.

        Returns the (replaced, source) pairs."""
        scores = [discounted_mean(p.history) if p.history else float("inf") for p in self.processes]
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        q = math.ceil(QUANTILE * len(order))
        top, bottom = order[:q], order[-q:]
        p_re = annealed(self.resample0, rnd, num_rounds)
        pairs = []
        for ib in bottom:
            it = top[int(self.rng.integers(len(top)))]
            src = self.processes[it]
            base = self.perturb(src.base, 2 * self.eps0, p_re)
            self.processes[ib] = Process(base=base, locals=self._locals_around(base, self.eps0), arrays=src.arrays)
            pairs.append((ib, it))
        for p in self.processes:
            p.history = []
        return pairs


def incumbent(round_losses: List[Optional[float]]) -> int:
    """Process reported in a round: the lowest pooled validation loss of that round (never the test)."""
    return min((i for i, v in enumerate(round_losses) if v is not None), key=lambda i: round_losses[i])
