from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .belief import active_belief_update, passive_belief_update
from .model import RMABInstance, StepResult


Policy = Callable[[tuple[np.ndarray, ...], RMABInstance, int, np.random.Generator], Sequence[int]]


class RMABSimulator:
    """Simulator for observe/reward-before-transition POMDP-RMAB dynamics."""

    def __init__(
        self,
        instance: RMABInstance,
        seed: int | None = None,
        initial_states: Sequence[int] | None = None,
    ) -> None:
        self.instance = instance
        self.rng = np.random.default_rng(seed)
        self.time = 0
        self.beliefs = tuple(b.copy() for b in instance.initial_beliefs)
        self.states = np.zeros(instance.num_arms, dtype=int)
        self.reset(initial_states=initial_states)

    def reset(self, initial_states: Sequence[int] | None = None) -> None:
        self.time = 0
        self.beliefs = tuple(b.copy() for b in self.instance.initial_beliefs)
        if initial_states is None:
            self.states = np.array(
                [
                    self.rng.choice(arm.num_states, p=self.beliefs[index])
                    for index, arm in enumerate(self.instance.arms)
                ],
                dtype=int,
            )
        else:
            states = np.asarray(initial_states, dtype=int)
            if states.shape != (self.instance.num_arms,):
                raise ValueError("initial_states must have shape (num_arms,).")
            for index, (state, arm) in enumerate(zip(states, self.instance.arms)):
                if state < 0 or state >= arm.num_states:
                    raise ValueError(f"initial_states[{index}] is out of range.")
            self.states = states.copy()

    def step(self, active_arms: Sequence[int]) -> StepResult:
        active_tuple = self._validate_active_arms(active_arms)
        active_set = set(active_tuple)

        previous_states = self.states.copy()
        next_states = np.empty_like(previous_states)
        rewards = np.zeros(self.instance.num_arms, dtype=float)
        observations: list[int | None] = []
        next_beliefs: list[np.ndarray] = []

        for arm_index, arm in enumerate(self.instance.arms):
            x = int(previous_states[arm_index])
            if arm_index in active_set:
                observation = int(self.rng.choice(arm.num_active_observations, p=arm.E1[x]))
                reward = float(arm.R[x, observation])
                next_belief, _ = active_belief_update(
                    self.beliefs[arm_index],
                    arm,
                    observation,
                    reward,
                )
                next_state = int(self.rng.choice(arm.num_states, p=arm.P1[x]))
            else:
                reward = 0.0
                if arm.E0 is None:
                    observation = None
                else:
                    observation = int(self.rng.choice(arm.num_passive_observations, p=arm.E0[x]))
                next_belief, _ = passive_belief_update(
                    self.beliefs[arm_index],
                    arm,
                    observation,
                )
                next_state = int(self.rng.choice(arm.num_states, p=arm.P0[x]))

            observations.append(observation)
            rewards[arm_index] = reward
            next_states[arm_index] = next_state
            next_beliefs.append(next_belief)

        self.states = next_states
        self.beliefs = tuple(next_beliefs)
        result = StepResult(
            time=self.time,
            active_arms=active_tuple,
            previous_states=previous_states,
            observations=tuple(observations),
            rewards_by_arm=rewards.copy(),
            total_reward=float(rewards.sum()),
            next_states=next_states.copy(),
            beliefs=tuple(b.copy() for b in self.beliefs),
        )
        self.time += 1
        return result

    def run(self, policy: Policy, steps: int | None = None) -> list[StepResult]:
        total_steps = self.instance.horizon if steps is None else int(steps)
        if total_steps < 0:
            raise ValueError("steps must be nonnegative.")
        results: list[StepResult] = []
        for _ in range(total_steps):
            action = policy(tuple(b.copy() for b in self.beliefs), self.instance, self.time, self.rng)
            results.append(self.step(action))
        return results

    def _validate_active_arms(self, active_arms: Sequence[int]) -> tuple[int, ...]:
        active_tuple = tuple(int(index) for index in active_arms)
        if len(set(active_tuple)) != len(active_tuple):
            raise ValueError("active arms must be unique.")
        if any(index < 0 or index >= self.instance.num_arms for index in active_tuple):
            raise ValueError("active arm index out of range.")
        if self.instance.exactly_k:
            if len(active_tuple) != self.instance.budget:
                raise ValueError(f"expected exactly K={self.instance.budget}, got {active_tuple}.")
        elif len(active_tuple) > self.instance.budget:
            raise ValueError(f"expected at most K={self.instance.budget}, got {active_tuple}.")
        return active_tuple


def cumulative_average_rewards(results: Sequence[StepResult]) -> np.ndarray:
    rewards = np.array([result.total_reward for result in results], dtype=float)
    if rewards.size == 0:
        return rewards
    return np.cumsum(rewards) / np.arange(1, rewards.size + 1)

