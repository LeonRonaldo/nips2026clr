from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryBatchInstance:
    """Vectorized two-state canonical RMAB for deployment experiments."""

    P0: np.ndarray
    P1: np.ndarray
    E1: np.ndarray
    R: np.ndarray
    initial_beliefs: np.ndarray
    budget: int
    horizon: int
    discount: float = 1.0
    E0: np.ndarray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        n = self.P0.shape[0]
        expected_matrix_shape = (n, 2, 2)
        for name in ("P0", "P1", "E1", "R"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != expected_matrix_shape:
                raise ValueError(f"{name} must have shape {expected_matrix_shape}, got {value.shape}.")
            object.__setattr__(self, name, value)
        if self.E0 is not None:
            e0 = np.asarray(self.E0, dtype=float)
            if e0.shape != expected_matrix_shape:
                raise ValueError(f"E0 must have shape {expected_matrix_shape}, got {e0.shape}.")
            object.__setattr__(self, "E0", e0)
        beliefs = np.asarray(self.initial_beliefs, dtype=float)
        if beliefs.shape != (n, 2):
            raise ValueError(f"initial_beliefs must have shape {(n, 2)}, got {beliefs.shape}.")
        object.__setattr__(self, "initial_beliefs", beliefs / beliefs.sum(axis=1, keepdims=True))
        if not 1 <= self.budget <= n:
            raise ValueError("budget must be between 1 and num_arms.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1].")

    @property
    def num_arms(self) -> int:
        return self.P0.shape[0]

    @property
    def has_passive_observation(self) -> bool:
        return self.E0 is not None


@dataclass(frozen=True)
class BatchStepResult:
    active_arms: np.ndarray
    reward: float
    cumulative_reward: float
    average_reward: float


class BinaryBatchSimulator:
    def __init__(
        self,
        instance: BinaryBatchInstance,
        seed: int,
        initial_states: np.ndarray | None = None,
    ) -> None:
        self.instance = instance
        self.rng = np.random.default_rng(seed)
        self.time = 0
        self.cumulative_reward = 0.0
        self.beliefs = instance.initial_beliefs.copy()
        if initial_states is None:
            self.states = _sample_states_from_beliefs(self.beliefs, self.rng)
        else:
            states = np.asarray(initial_states, dtype=int)
            if states.shape != (instance.num_arms,):
                raise ValueError("initial_states has wrong shape.")
            self.states = states.copy()

    def step(self, active_arms: Sequence[int]) -> BatchStepResult:
        active = _validate_action(active_arms, self.instance)
        reward, self.states, self.beliefs = sample_public_step(
            self.instance,
            self.beliefs,
            self.states,
            active,
            self.rng,
        )
        self.cumulative_reward += reward
        self.time += 1
        return BatchStepResult(
            active_arms=active,
            reward=reward,
            cumulative_reward=self.cumulative_reward,
            average_reward=self.cumulative_reward / self.time,
        )


class MyopicBatchPolicy:
    def __call__(
        self,
        beliefs: np.ndarray,
        instance: BinaryBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del time, rng
        return top_k(expected_active_rewards(instance, beliefs), instance.budget)


class SimpleVFABatchPolicy:
    """Hand-built deployable VFA baseline.

    It scores active-minus-passive using a one-step lookahead into a simple
    approximate continuation value. No true hidden state is used.
    """

    def __init__(
        self,
        continuation_weight: float = 0.75,
        information_weight: float = 0.05,
        persistence_weight: float = 0.08,
    ) -> None:
        self.continuation_weight = float(continuation_weight)
        self.information_weight = float(information_weight)
        self.persistence_weight = float(persistence_weight)

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: BinaryBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del time, rng
        scores = np.empty(instance.num_arms, dtype=float)
        for arm_index in range(instance.num_arms):
            q_active = expected_active_rewards(instance, beliefs, arm_index) + instance.discount * (
                self._expected_feature_value(instance, beliefs[arm_index], arm_index, active=True)
            )
            q_passive = instance.discount * self._expected_feature_value(
                instance,
                beliefs[arm_index],
                arm_index,
                active=False,
            )
            scores[arm_index] = q_active - q_passive
        return top_k(scores, instance.budget)

    def _feature_value(self, instance: BinaryBatchInstance, belief: np.ndarray, arm_index: int) -> float:
        reward = expected_active_rewards(instance, belief.reshape(1, 2), 0, source_arm=arm_index)
        p_good = float(belief[1])
        entropy_bonus = 1.0 - _binary_entropy(p_good)
        persistence = float(instance.P1[arm_index, 1, 1] - instance.P0[arm_index, 1, 1])
        return (
            self.continuation_weight * reward
            + self.information_weight * entropy_bonus
            + self.persistence_weight * p_good * persistence
        )

    def _expected_feature_value(
        self,
        instance: BinaryBatchInstance,
        belief: np.ndarray,
        arm_index: int,
        *,
        active: bool,
    ) -> float:
        total = 0.0
        for probability, next_belief in public_feedback_outcomes(
            instance,
            belief,
            arm_index,
            active=active,
        ):
            total += probability * self._feature_value(instance, next_belief, arm_index)
        return float(total)


class RolloutMyopicBatchPolicy:
    """Short simulation rollout around myopic candidate actions."""

    def __init__(
        self,
        rollout_horizon: int = 2,
        rollout_samples: int = 2,
        candidate_actions: int = 4,
        seed: int = 0,
    ) -> None:
        if rollout_horizon <= 0:
            raise ValueError("rollout_horizon must be positive.")
        if rollout_samples <= 0:
            raise ValueError("rollout_samples must be positive.")
        if candidate_actions <= 0:
            raise ValueError("candidate_actions must be positive.")
        self.rollout_horizon = int(rollout_horizon)
        self.rollout_samples = int(rollout_samples)
        self.candidate_actions = int(candidate_actions)
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: BinaryBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del time, rng
        scores = expected_active_rewards(instance, beliefs)
        candidates = self._candidate_actions(instance, beliefs, scores)
        if len(candidates) == 1:
            return candidates[0]
        seeds = [int(self.rng.integers(0, 2**32 - 1)) for _ in range(self.rollout_samples)]
        values = np.array(
            [self._estimate_candidate(instance, beliefs, candidate, seeds) for candidate in candidates],
            dtype=float,
        )
        return candidates[int(np.argmax(values))]

    def _candidate_actions(
        self,
        instance: BinaryBatchInstance,
        beliefs: np.ndarray,
        scores: np.ndarray,
    ) -> list[np.ndarray]:
        base = top_k(scores, instance.budget)
        candidates = [base]
        if self.candidate_actions == 1:
            return candidates

        base_set = set(int(index) for index in base)
        entropy = _binary_entropy_vec(beliefs[:, 1])
        drop_order = [int(index) for index in base[np.argsort(scores[base], kind="mergesort")]]
        add_pool: list[int] = []
        for order in (np.argsort(-entropy, kind="mergesort"), np.argsort(-scores, kind="mergesort")):
            for index in order:
                arm = int(index)
                if arm not in base_set and arm not in add_pool:
                    add_pool.append(arm)
                if len(add_pool) >= self.candidate_actions * 2:
                    break
            if len(add_pool) >= self.candidate_actions * 2:
                break

        for add_arm in add_pool:
            for drop_arm in drop_order:
                trial_set = (base_set - {drop_arm}) | {add_arm}
                trial = np.array(sorted(trial_set), dtype=int)
                if not any(np.array_equal(trial, candidate) for candidate in candidates):
                    candidates.append(trial)
                    if len(candidates) >= self.candidate_actions:
                        return candidates
        return candidates

    def _estimate_candidate(
        self,
        instance: BinaryBatchInstance,
        beliefs: np.ndarray,
        first_action: np.ndarray,
        seeds: Sequence[int],
    ) -> float:
        values = [
            self._sample_rollout_return(instance, beliefs, first_action, int(seed))
            for seed in seeds
        ]
        return float(np.mean(values))

    def _sample_rollout_return(
        self,
        instance: BinaryBatchInstance,
        beliefs: np.ndarray,
        first_action: np.ndarray,
        seed: int,
    ) -> float:
        rng = np.random.default_rng(seed)
        sim_beliefs = beliefs.copy()
        states = _sample_states_from_beliefs(sim_beliefs, rng)
        total = 0.0
        discount = 1.0
        for step in range(self.rollout_horizon):
            if step == 0:
                action = first_action
            else:
                action = top_k(expected_active_rewards(instance, sim_beliefs), instance.budget)
            reward, states, sim_beliefs = sample_public_step(
                instance,
                sim_beliefs,
                states,
                action,
                rng,
            )
            total += discount * reward
            discount *= instance.discount
        return float(total)


def make_random_binary_instance(
    *,
    num_arms: int,
    budget: int,
    horizon: int,
    seed: int,
    case: str = "case3",
    discount: float = 1.0,
) -> BinaryBatchInstance:
    if case not in {"case2", "case3"}:
        raise ValueError("large-scale generator currently supports case2 or case3.")
    rng = np.random.default_rng(seed)
    p01_passive = rng.uniform(0.01, 0.08, size=num_arms)
    p11_passive = rng.uniform(0.82, 0.98, size=num_arms)
    p01_active = np.clip(p01_passive + rng.uniform(0.00, 0.04, size=num_arms), 0.0, 0.25)
    p11_active = np.clip(p11_passive + rng.uniform(0.00, 0.03, size=num_arms), 0.0, 0.995)

    P0 = np.zeros((num_arms, 2, 2), dtype=float)
    P1 = np.zeros((num_arms, 2, 2), dtype=float)
    P0[:, 0, 1] = p01_passive
    P0[:, 0, 0] = 1.0 - p01_passive
    P0[:, 1, 1] = p11_passive
    P0[:, 1, 0] = 1.0 - p11_passive
    P1[:, 0, 1] = p01_active
    P1[:, 0, 0] = 1.0 - p01_active
    P1[:, 1, 1] = p11_active
    P1[:, 1, 0] = 1.0 - p11_active

    active_accuracy = rng.uniform(0.82, 0.94, size=num_arms)
    E1 = _symmetric_binary_observation(active_accuracy)
    E0 = None
    if case == "case3":
        passive_accuracy = rng.uniform(0.52, 0.64, size=num_arms)
        E0 = _symmetric_binary_observation(passive_accuracy)

    scale = rng.uniform(0.65, 1.35, size=num_arms)
    R = np.zeros((num_arms, 2, 2), dtype=float)
    R[:, 0, 0] = rng.uniform(0.00, 0.04, size=num_arms) * scale
    R[:, 0, 1] = rng.uniform(0.00, 0.02, size=num_arms) * scale
    R[:, 1, 0] = rng.uniform(0.45, 0.72, size=num_arms) * scale
    R[:, 1, 1] = rng.uniform(0.86, 1.08, size=num_arms) * scale

    initial_good = rng.beta(2.0, 5.0, size=num_arms)
    initial_beliefs = np.column_stack([1.0 - initial_good, initial_good])
    return BinaryBatchInstance(
        P0=P0,
        P1=P1,
        E0=E0,
        E1=E1,
        R=R,
        initial_beliefs=initial_beliefs,
        budget=budget,
        horizon=horizon,
        discount=discount,
        name=f"{case}_binary_N{num_arms}_K{budget}_T{horizon}",
    )


def sample_public_step(
    instance: BinaryBatchInstance,
    beliefs: np.ndarray,
    states: np.ndarray,
    active_arms: Sequence[int],
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    n = instance.num_arms
    active = _validate_action(active_arms, instance)
    active_mask = np.zeros(n, dtype=bool)
    active_mask[active] = True
    passive_mask = ~active_mask
    next_beliefs = np.empty_like(beliefs)
    next_states = np.empty_like(states)
    rewards = np.zeros(n, dtype=float)

    if active.size:
        idx = active
        x = states[idx]
        obs_prob_one = instance.E1[idx, x, 1]
        obs = (rng.random(idx.size) < obs_prob_one).astype(int)
        rewards[idx] = instance.R[idx, x, obs]
        next_beliefs[idx] = _active_next_beliefs(instance, beliefs[idx], idx, obs, rewards[idx])
        next_prob_one = instance.P1[idx, x, 1]
        next_states[idx] = (rng.random(idx.size) < next_prob_one).astype(int)

    passive_idx = np.flatnonzero(passive_mask)
    if passive_idx.size:
        x = states[passive_idx]
        if instance.E0 is None:
            next_beliefs[passive_idx] = _matmul_rows(beliefs[passive_idx], instance.P0[passive_idx])
        else:
            obs_prob_one = instance.E0[passive_idx, x, 1]
            obs = (rng.random(passive_idx.size) < obs_prob_one).astype(int)
            next_beliefs[passive_idx] = _passive_next_beliefs(instance, beliefs[passive_idx], passive_idx, obs)
        next_prob_one = instance.P0[passive_idx, x, 1]
        next_states[passive_idx] = (rng.random(passive_idx.size) < next_prob_one).astype(int)

    return float(rewards.sum()), next_states, next_beliefs


def expected_active_rewards(
    instance: BinaryBatchInstance,
    beliefs: np.ndarray,
    arm_index: int | None = None,
    *,
    source_arm: int | None = None,
) -> np.ndarray | float:
    if arm_index is not None:
        source = arm_index if source_arm is None else source_arm
        state_rewards = np.sum(instance.E1[source] * instance.R[source], axis=1)
        return float(np.asarray(beliefs, dtype=float)[arm_index] @ state_rewards)
    state_rewards = np.sum(instance.E1 * instance.R, axis=2)
    return np.sum(np.asarray(beliefs, dtype=float) * state_rewards, axis=1)


def public_feedback_outcomes(
    instance: BinaryBatchInstance,
    belief: np.ndarray,
    arm_index: int,
    *,
    active: bool,
) -> tuple[tuple[float, np.ndarray], ...]:
    b = np.asarray(belief, dtype=float)
    if active:
        outcomes: list[tuple[float, np.ndarray]] = []
        for obs in (0, 1):
            for reward in sorted({float(instance.R[arm_index, 0, obs]), float(instance.R[arm_index, 1, obs])}):
                likelihood = instance.E1[arm_index, :, obs] * np.isclose(
                    instance.R[arm_index, :, obs],
                    reward,
                    atol=1e-12,
                    rtol=0.0,
                )
                prob = float(b @ likelihood)
                if prob <= 1e-14:
                    continue
                posterior = b * likelihood / prob
                next_belief = posterior @ instance.P1[arm_index]
                outcomes.append((prob, _normalize(next_belief)))
        return tuple(outcomes)

    if instance.E0 is None:
        return ((1.0, _normalize(b @ instance.P0[arm_index])),)
    outcomes = []
    for obs in (0, 1):
        likelihood = instance.E0[arm_index, :, obs]
        prob = float(b @ likelihood)
        if prob <= 1e-14:
            continue
        posterior = b * likelihood / prob
        next_belief = posterior @ instance.P0[arm_index]
        outcomes.append((prob, _normalize(next_belief)))
    return tuple(outcomes)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.size:
        return np.arange(scores.size, dtype=int)
    partition = np.argpartition(-scores, k - 1)[:k]
    return partition[np.argsort(-scores[partition], kind="mergesort")].astype(int)


def _active_next_beliefs(
    instance: BinaryBatchInstance,
    beliefs: np.ndarray,
    idx: np.ndarray,
    observations: np.ndarray,
    rewards: np.ndarray,
) -> np.ndarray:
    likelihood0 = instance.E1[idx, 0, observations] * np.isclose(
        instance.R[idx, 0, observations],
        rewards,
        atol=1e-12,
        rtol=0.0,
    )
    likelihood1 = instance.E1[idx, 1, observations] * np.isclose(
        instance.R[idx, 1, observations],
        rewards,
        atol=1e-12,
        rtol=0.0,
    )
    denom = beliefs[:, 0] * likelihood0 + beliefs[:, 1] * likelihood1
    posterior = np.column_stack([
        beliefs[:, 0] * likelihood0 / denom,
        beliefs[:, 1] * likelihood1 / denom,
    ])
    return _matmul_rows(posterior, instance.P1[idx])


def _passive_next_beliefs(
    instance: BinaryBatchInstance,
    beliefs: np.ndarray,
    idx: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    likelihood0 = instance.E0[idx, 0, observations]
    likelihood1 = instance.E0[idx, 1, observations]
    denom = beliefs[:, 0] * likelihood0 + beliefs[:, 1] * likelihood1
    posterior = np.column_stack([
        beliefs[:, 0] * likelihood0 / denom,
        beliefs[:, 1] * likelihood1 / denom,
    ])
    return _matmul_rows(posterior, instance.P0[idx])


def _matmul_rows(beliefs: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    out0 = beliefs[:, 0] * matrices[:, 0, 0] + beliefs[:, 1] * matrices[:, 1, 0]
    out1 = beliefs[:, 0] * matrices[:, 0, 1] + beliefs[:, 1] * matrices[:, 1, 1]
    return np.column_stack([out0, out1])


def _sample_states_from_beliefs(beliefs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(beliefs.shape[0]) < beliefs[:, 1]).astype(int)


def _symmetric_binary_observation(accuracy: np.ndarray) -> np.ndarray:
    matrix = np.zeros((accuracy.size, 2, 2), dtype=float)
    matrix[:, 0, 0] = accuracy
    matrix[:, 0, 1] = 1.0 - accuracy
    matrix[:, 1, 0] = 1.0 - accuracy
    matrix[:, 1, 1] = accuracy
    return matrix


def _validate_action(active_arms: Sequence[int], instance: BinaryBatchInstance) -> np.ndarray:
    active = np.asarray(active_arms, dtype=int)
    if active.shape != (instance.budget,):
        raise ValueError(f"expected exactly {instance.budget} active arms, got shape {active.shape}.")
    if np.unique(active).size != active.size:
        raise ValueError("active arms must be unique.")
    if np.any(active < 0) or np.any(active >= instance.num_arms):
        raise ValueError("active arm index out of range.")
    return active


def _binary_entropy(p: float) -> float:
    clipped = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return float(-(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)))


def _binary_entropy_vec(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def _normalize(belief: np.ndarray) -> np.ndarray:
    b = np.maximum(np.asarray(belief, dtype=float), 0.0)
    return b / b.sum()

