"""Shared step-size tuning for HMC samplers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from tqdm import tqdm


State = TypeVar("State")


def tune_step_size(
    initial_state: State,
    initial_step_size: float,
    metropolis_step: Callable[[State, float], tuple[State, bool, float]],
    *,
    n_tune_steps: int,
    target_rate: float = 0.70,
    target_tolerance: float = 0.15,
    max_attempts: int = 10,
    step_min: float = 1e-6,
    step_max: float = 1.0,
    description: str = "Tuning step size",
) -> float:
    """Tune an HMC step size with the repository's historical binary search."""
    if n_tune_steps <= 0:
        raise ValueError("n_tune_steps must be positive when step-size tuning is enabled")
    if not 0.0 < target_rate < 1.0:
        raise ValueError("target_rate must be between 0 and 1")
    if not 0.0 <= target_tolerance < 1.0:
        raise ValueError("target_tolerance must be in [0, 1)")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if not 0.0 < step_min < step_max:
        raise ValueError("step-size bounds must satisfy 0 < step_min < step_max")

    current_dt = float(initial_step_size)
    if not step_min <= current_dt <= step_max:
        raise ValueError(f"initial step size must be in [{step_min}, {step_max}]")

    state = initial_state
    best_dt = current_dt
    best_rate_diff = float("inf")

    for attempt in range(max_attempts):
        acceptance_count = 0
        for _ in tqdm(range(n_tune_steps), desc=f"{description} ({attempt + 1}/{max_attempts})"):
            state, accepted, _ = metropolis_step(state, current_dt)
            acceptance_count += int(accepted)

        acceptance_rate = acceptance_count / n_tune_steps
        rate_diff = abs(acceptance_rate - target_rate)
        print(f"Step size: {current_dt:.6f}, acceptance rate: {acceptance_rate:.2%}")
        if rate_diff < best_rate_diff:
            best_dt = current_dt
            best_rate_diff = rate_diff
        if rate_diff <= target_tolerance:
            return current_dt

        if acceptance_rate > target_rate:
            step_min = current_dt
            current_dt = min((current_dt + step_max) / 2, step_max)
        else:
            step_max = current_dt
            current_dt = max((current_dt + step_min) / 2, step_min)

    return best_dt
