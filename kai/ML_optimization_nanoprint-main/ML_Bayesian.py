"""Bayesian optimization helpers for selecting the next experiment parameters."""

from __future__ import annotations

import random
from typing import Any

from config import MIN_HISTORY_FOR_BO, OPTIMIZER_RANDOM_STATE, PARAMETER_BOUNDS
from utils import history_target_from_row, read_history_rows, safe_float

try:
    from bayes_opt import BayesianOptimization
except ImportError:  # pragma: no cover - dependency may be missing locally
    BayesianOptimization = None


def _fallback_suggestion(history_rows: list[dict[str, str]]) -> dict[str, float]:
    """Fallback suggestion when bayesian-optimization is unavailable."""

    rng = random.Random(OPTIMIZER_RANDOM_STATE + len(history_rows))

    if not history_rows:
        return {
            "mix_ratio": sum(PARAMETER_BOUNDS["mix_ratio"]) / 2.0,
            "mix_time": sum(PARAMETER_BOUNDS["mix_time"]) / 2.0,
        }

    best_row = max(history_rows, key=history_target_from_row)
    best_mix_ratio = safe_float(best_row.get("mix_ratio"), sum(PARAMETER_BOUNDS["mix_ratio"]) / 2.0)
    best_mix_time = safe_float(best_row.get("mix_time"), sum(PARAMETER_BOUNDS["mix_time"]) / 2.0)

    ratio_jitter = 0.08 * (PARAMETER_BOUNDS["mix_ratio"][1] - PARAMETER_BOUNDS["mix_ratio"][0])
    time_jitter = 0.08 * (PARAMETER_BOUNDS["mix_time"][1] - PARAMETER_BOUNDS["mix_time"][0])

    return {
        "mix_ratio": min(
            PARAMETER_BOUNDS["mix_ratio"][1],
            max(PARAMETER_BOUNDS["mix_ratio"][0], best_mix_ratio + rng.uniform(-ratio_jitter, ratio_jitter)),
        ),
        "mix_time": min(
            PARAMETER_BOUNDS["mix_time"][1],
            max(PARAMETER_BOUNDS["mix_time"][0], best_mix_time + rng.uniform(-time_jitter, time_jitter)),
        ),
    }


def _suggest_with_bayes_opt(history_rows: list[dict[str, str]]) -> dict[str, float]:
    """Use Bayesian Optimization to maximize ``final_grade``."""

    optimizer = BayesianOptimization(
        f=None,
        pbounds=PARAMETER_BOUNDS,
        random_state=OPTIMIZER_RANDOM_STATE,
        verbose=0,
    )

    for row in history_rows:
        optimizer.register(
            params={
                "mix_ratio": safe_float(row.get("mix_ratio"), 0.0),
                "mix_time": safe_float(row.get("mix_time"), 0.0),
            },
            target=history_target_from_row(row),
        )

    try:
        next_point: dict[str, Any] = optimizer.suggest()
    except TypeError:  # pragma: no cover - compatibility shim for older releases
        try:
            from bayes_opt import UtilityFunction
        except ImportError:
            from bayes_opt.util import UtilityFunction

        utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        next_point = optimizer.suggest(utility)

    return {
        "mix_ratio": float(next_point["mix_ratio"]),
        "mix_time": float(next_point["mix_time"]),
    }


def get_next_parameters(history_file: str) -> dict[str, float]:
    """Return the next ``mix_ratio`` and ``mix_time`` to try.

    The optimizer is configured to maximize the scalar target ``final_grade``.
    """

    history_rows = read_history_rows(history_file)

    if BayesianOptimization is None or len(history_rows) < MIN_HISTORY_FOR_BO:
        next_point = _fallback_suggestion(history_rows)
    else:
        next_point = _suggest_with_bayes_opt(history_rows)

    print("\nLoaded previous experiments.")
    print("Optimizer objective: maximize final_grade")
    print("\nNext suggested parameters:")
    print(f"mix_ratio = {next_point['mix_ratio']:.4f}")
    print(f"mix_time = {next_point['mix_time']:.4f}")
    return next_point
