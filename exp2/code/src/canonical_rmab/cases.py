from __future__ import annotations

import numpy as np

from .model import ArmModel, RMABInstance


def make_case1(num_arms: int = 2, budget: int = 1, horizon: int = 20) -> RMABInstance:
    p0 = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=float)
    p1 = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    e1 = np.eye(2, dtype=float)
    reward = np.array([[0.0, -1.0], [-1.0, 1.0]], dtype=float)
    return _repeat_arm_instance(
        ArmModel(P0=p0, P1=p1, E1=e1, E0=None, R=reward, name="case1_arm"),
        num_arms=num_arms,
        budget=budget,
        horizon=horizon,
        name="case1_active_exact",
    )


def make_case2(num_arms: int = 2, budget: int = 1, horizon: int = 20) -> RMABInstance:
    p0 = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=float)
    p1 = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=float)
    e1 = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    reward = np.array([[10.0, 1.0], [20.0, 2.0]], dtype=float)
    return _repeat_arm_instance(
        ArmModel(P0=p0, P1=p1, E1=e1, E0=None, R=reward, name="case2_arm"),
        num_arms=num_arms,
        budget=budget,
        horizon=horizon,
        name="case2_active_noisy",
    )


def make_case3(num_arms: int = 2, budget: int = 1, horizon: int = 20) -> RMABInstance:
    p0 = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    p1 = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=float)
    e0 = np.array([[0.6, 0.4], [0.45, 0.55]], dtype=float)
    e1 = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    reward = np.array([[10.0, 1.0], [20.0, 2.0]], dtype=float)
    return _repeat_arm_instance(
        ArmModel(P0=p0, P1=p1, E1=e1, E0=e0, R=reward, name="case3_arm"),
        num_arms=num_arms,
        budget=budget,
        horizon=horizon,
        name="case3_weak_strong_noisy",
    )


def _repeat_arm_instance(
    arm: ArmModel,
    *,
    num_arms: int,
    budget: int,
    horizon: int,
    name: str,
) -> RMABInstance:
    if num_arms <= 0:
        raise ValueError("num_arms must be positive.")
    beliefs = tuple(np.full(arm.num_states, 1.0 / arm.num_states) for _ in range(num_arms))
    arms = tuple(
        ArmModel(
            P0=arm.P0.copy(),
            P1=arm.P1.copy(),
            E0=None if arm.E0 is None else arm.E0.copy(),
            E1=arm.E1.copy(),
            R=arm.R.copy(),
            name=f"{arm.name}_{index}",
        )
        for index in range(num_arms)
    )
    return RMABInstance(
        arms=arms,
        budget=budget,
        initial_beliefs=beliefs,
        horizon=horizon,
        discount=1.0,
        exactly_k=True,
        name=name,
    )

