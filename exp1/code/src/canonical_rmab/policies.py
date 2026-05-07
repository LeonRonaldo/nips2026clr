from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .belief import expected_active_reward
from .model import RMABInstance


def top_k_indices(scores: np.ndarray, k: int) -> tuple[int, ...]:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    return tuple(int(index) for index in order[:k])


def myopic_policy(
    beliefs: tuple[np.ndarray, ...],
    instance: RMABInstance,
    time: int,
    rng: np.random.Generator,
) -> Sequence[int]:
    del time, rng
    scores = np.array(
        [
            expected_active_reward(belief, arm)
            for belief, arm in zip(beliefs, instance.arms)
        ],
        dtype=float,
    )
    return top_k_indices(scores, instance.budget)

