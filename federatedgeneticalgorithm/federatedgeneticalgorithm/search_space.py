"""The harmful region of the HP grid, and how a searcher is kept out of it.

FedGA and RS redraw a proposal that falls in it (draw_allowed); TPE samples lr from the rates each
optimizer has left (allowed_lrs_by_optimizer).

Harmful = lion with lr >= 5e-3, or adam/adamw/radam with lr >= 1e-2: 40 of the 320 behaviorally
distinct configurations, the set the paper reads off the broadcast drops of seeds 0-4.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Dict, List, Optional

HARMFUL_LR_FLOOR = {"lion": 5e-3, "adam": 1e-2, "adamw": 1e-2, "radam": 1e-2}

# a proposal is redrawn this many times at most; the allowed region is 7/8 of the grid
MAX_REDRAWS = 10_000


def is_harmful(hp: Dict) -> bool:
    floor = HARMFUL_LR_FLOOR.get(hp["optimizer"])
    return floor is not None and float(hp["lr"]) >= floor


def draw_allowed(make: Callable[[], Dict], exclude: Optional[Callable[[Dict], bool]]) -> tuple[Dict, int]:
    """Calls make() until exclude() rejects nothing; returns the HP and how many draws were discarded.

    With exclude=None make() runs exactly once, so the searcher's RNG stream is the unpruned one.
    """
    hp = make()
    redraws = 0
    while exclude is not None and exclude(hp):
        redraws += 1
        if redraws > MAX_REDRAWS:
            raise RuntimeError("no allowed HP after MAX_REDRAWS draws; is the whole grid excluded?")
        hp = make()
    return hp, redraws


def allowed_lrs_by_optimizer(hyperparams: Dict[str, List], exclude: Callable[[Dict], bool]) -> Dict[str, List]:
    """Learning rates left to each optimizer; exclude must depend on optimizer and lr only."""
    others = [k for k in hyperparams if k not in ("optimizers", "learning_rates")]
    singular = {"batch_sizes": "batch_size", "weight_decays": "weight_decay", "momentums": "momentum"}
    allowed: Dict[str, List] = {}
    for opt in hyperparams["optimizers"]:
        allowed[opt] = []
        for lr in hyperparams["learning_rates"]:
            verdicts = {
                exclude({"optimizer": opt, "lr": lr, **{singular[k]: v for k, v in zip(others, combo)}})
                for combo in product(*(hyperparams[k] for k in others))
            }
            if len(verdicts) > 1:
                raise ValueError(f"exclude depends on more than optimizer and lr ({opt}, lr={lr})")
            if not verdicts.pop():
                allowed[opt].append(lr)
        if not allowed[opt]:
            raise ValueError(f"exclude leaves no learning rate for {opt}")
    return allowed
