from __future__ import annotations

import numpy as np

from .model import ArmModel
from .validation import normalize_probability_vector, validate_probability_vector


def active_belief_update(
    belief: np.ndarray,
    arm: ArmModel,
    observation: int,
    reward: float,
    reward_tol: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Bayes update on `(O,Y)` for current state, then prediction through `P1`."""
    b = validate_probability_vector(belief, "belief")
    if not 0 <= int(observation) < arm.num_active_observations:
        raise ValueError("active observation is out of range.")
    o = int(observation)
    compatible_reward = np.isclose(arm.R[:, o], float(reward), atol=reward_tol, rtol=0.0)
    likelihood = arm.E1[:, o] * compatible_reward.astype(float)
    feedback_prob = float(b @ likelihood)
    if feedback_prob <= 0.0:
        raise ValueError("active feedback has zero probability under current belief.")
    posterior_current = b * likelihood / feedback_prob
    next_belief = posterior_current @ arm.P1
    return normalize_probability_vector(next_belief, "active_next_belief"), feedback_prob


def passive_belief_update(
    belief: np.ndarray,
    arm: ArmModel,
    observation: int | None = None,
) -> tuple[np.ndarray, float]:
    """Passive update: optional `E0` signal, zero reward, then prediction through `P0`."""
    b = validate_probability_vector(belief, "belief")
    if arm.E0 is None:
        if observation is not None:
            raise ValueError("passive observation supplied but arm has no E0.")
        posterior_current = b
        feedback_prob = 1.0
    else:
        if observation is None:
            raise ValueError("passive observation is required when E0 exists.")
        if not 0 <= int(observation) < arm.num_passive_observations:
            raise ValueError("passive observation is out of range.")
        likelihood = arm.E0[:, int(observation)]
        feedback_prob = float(b @ likelihood)
        if feedback_prob <= 0.0:
            raise ValueError("passive feedback has zero probability under current belief.")
        posterior_current = b * likelihood / feedback_prob
    next_belief = posterior_current @ arm.P0
    return normalize_probability_vector(next_belief, "passive_next_belief"), feedback_prob


def expected_active_reward(belief: np.ndarray, arm: ArmModel) -> float:
    b = validate_probability_vector(belief, "belief")
    state_rewards = np.sum(arm.E1 * arm.R, axis=1)
    return float(b @ state_rewards)


def expected_passive_reward(belief: np.ndarray, arm: ArmModel) -> float:
    validate_probability_vector(belief, "belief")
    del arm
    return 0.0

