from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .validation import (
    as_float_array,
    validate_probability_vector,
    validate_row_stochastic,
)


@dataclass(frozen=True)
class ArmModel:
    """One restless POMDP arm under the canonical observe-then-transition model."""

    P0: np.ndarray
    P1: np.ndarray
    E1: np.ndarray
    R: np.ndarray
    E0: np.ndarray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        p0 = validate_row_stochastic(self.P0, "P0")
        p1 = validate_row_stochastic(self.P1, "P1")
        if p0.shape[0] != p0.shape[1]:
            raise ValueError("P0 must be square.")
        if p1.shape != p0.shape:
            raise ValueError("P1 must have the same square shape as P0.")
        d = p0.shape[0]

        e1 = validate_row_stochastic(self.E1, "E1")
        if e1.shape[0] != d:
            raise ValueError("E1 must have one row per hidden state.")

        reward = as_float_array(self.R, "R", ndim=2)
        if reward.shape != e1.shape:
            raise ValueError(f"R must have shape {e1.shape}, got {reward.shape}.")

        e0 = None
        if self.E0 is not None:
            e0 = validate_row_stochastic(self.E0, "E0")
            if e0.shape[0] != d:
                raise ValueError("E0 must have one row per hidden state.")

        object.__setattr__(self, "P0", p0)
        object.__setattr__(self, "P1", p1)
        object.__setattr__(self, "E0", e0)
        object.__setattr__(self, "E1", e1)
        object.__setattr__(self, "R", reward)

    @property
    def num_states(self) -> int:
        return self.P0.shape[0]

    @property
    def num_active_observations(self) -> int:
        return self.E1.shape[1]

    @property
    def num_passive_observations(self) -> int:
        return 0 if self.E0 is None else self.E0.shape[1]


@dataclass(frozen=True)
class RMABInstance:
    arms: tuple[ArmModel, ...]
    budget: int
    initial_beliefs: tuple[np.ndarray, ...]
    horizon: int
    discount: float = 1.0
    exactly_k: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        arms = tuple(self.arms)
        if not arms:
            raise ValueError("RMABInstance needs at least one arm.")
        if self.budget < 0 or self.budget > len(arms):
            raise ValueError("budget must be between 0 and the number of arms.")
        if self.exactly_k and self.budget == 0:
            raise ValueError("exactly-K instances need positive budget.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1].")

        beliefs = _coerce_initial_beliefs(self.initial_beliefs, arms)
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "initial_beliefs", beliefs)

    @property
    def num_arms(self) -> int:
        return len(self.arms)


@dataclass(frozen=True)
class StepResult:
    time: int
    active_arms: tuple[int, ...]
    previous_states: np.ndarray
    observations: tuple[int | None, ...]
    rewards_by_arm: np.ndarray
    total_reward: float
    next_states: np.ndarray
    beliefs: tuple[np.ndarray, ...]


def _coerce_initial_beliefs(
    initial_beliefs: Sequence[object] | np.ndarray,
    arms: tuple[ArmModel, ...],
) -> tuple[np.ndarray, ...]:
    if isinstance(initial_beliefs, np.ndarray) and initial_beliefs.ndim == 2:
        if initial_beliefs.shape[0] != len(arms):
            raise ValueError("initial_beliefs row count must equal number of arms.")
        raw = [initial_beliefs[i] for i in range(initial_beliefs.shape[0])]
    else:
        raw = list(initial_beliefs)
        if len(raw) != len(arms):
            raise ValueError("initial_beliefs length must equal number of arms.")

    beliefs: list[np.ndarray] = []
    for index, (belief, arm) in enumerate(zip(raw, arms)):
        vector = validate_probability_vector(belief, f"initial_beliefs[{index}]")
        if vector.shape != (arm.num_states,):
            raise ValueError(
                f"initial_beliefs[{index}] shape must be {(arm.num_states,)}, got {vector.shape}."
            )
        beliefs.append(vector)
    return tuple(beliefs)

