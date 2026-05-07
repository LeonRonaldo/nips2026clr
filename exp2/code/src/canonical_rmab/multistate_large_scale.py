from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_STATE_QUALITY_CACHE: dict[int, np.ndarray] = {}
_INSTANCE_CACHE_KEY = tuple[int, int, int, int, str]
_STATE_REWARD_CACHE: dict[_INSTANCE_CACHE_KEY, np.ndarray] = {}
_ACTIVE_FEEDBACK_LIKELIHOOD_CACHE: dict[tuple[_INSTANCE_CACHE_KEY, int], tuple[np.ndarray, ...]] = {}
_PASSIVE_FEEDBACK_LIKELIHOOD_CACHE: dict[tuple[_INSTANCE_CACHE_KEY, int], tuple[np.ndarray, ...]] = {}


@dataclass(frozen=True)
class MultiStateBatchInstance:
    """Vectorized canonical RMAB instance with a shared state/observation count.

    This module is intentionally independent from the old project code. It uses
    the canonical observation/reward-before-transition convention and keeps the
    realized reward as R[true_state, observation].
    """

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
        p0 = _as_probability_tensor(self.P0, "P0")
        p1 = _as_probability_tensor(self.P1, "P1")
        if p0.shape != p1.shape:
            raise ValueError(f"P1 must have shape {p0.shape}, got {p1.shape}.")
        if p0.shape[1] != p0.shape[2]:
            raise ValueError("P0 and P1 must be square per arm.")

        e1 = _as_probability_tensor(self.E1, "E1")
        if e1.shape[:2] != p0.shape[:2]:
            raise ValueError(f"E1 must start with shape {p0.shape[:2]}, got {e1.shape}.")

        reward = np.asarray(self.R, dtype=float)
        if reward.shape != e1.shape:
            raise ValueError(f"R must have shape {e1.shape}, got {reward.shape}.")

        e0 = None
        if self.E0 is not None:
            e0 = _as_probability_tensor(self.E0, "E0")
            if e0.shape[:2] != p0.shape[:2]:
                raise ValueError(f"E0 must start with shape {p0.shape[:2]}, got {e0.shape}.")
            if e0.shape[2] != e1.shape[2]:
                raise ValueError("This batch implementation expects E0 and E1 to share observation count.")

        beliefs = np.asarray(self.initial_beliefs, dtype=float)
        expected_belief_shape = (p0.shape[0], p0.shape[1])
        if beliefs.shape != expected_belief_shape:
            raise ValueError(f"initial_beliefs must have shape {expected_belief_shape}, got {beliefs.shape}.")
        if np.any(beliefs < 0.0):
            raise ValueError("initial_beliefs must be nonnegative.")
        row_sums = beliefs.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("initial_beliefs rows must have positive mass.")

        if not 1 <= self.budget <= p0.shape[0]:
            raise ValueError("budget must be between 1 and num_arms.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1].")

        object.__setattr__(self, "P0", p0)
        object.__setattr__(self, "P1", p1)
        object.__setattr__(self, "E1", e1)
        object.__setattr__(self, "E0", e0)
        object.__setattr__(self, "R", reward)
        object.__setattr__(self, "initial_beliefs", beliefs / row_sums)

    @property
    def num_arms(self) -> int:
        return self.P0.shape[0]

    @property
    def num_states(self) -> int:
        return self.P0.shape[1]

    @property
    def num_observations(self) -> int:
        return self.E1.shape[2]

    @property
    def has_passive_observation(self) -> bool:
        return self.E0 is not None


@dataclass(frozen=True)
class MultiStateStepResult:
    active_arms: np.ndarray
    reward: float
    cumulative_reward: float
    average_reward: float


class MultiStateBatchSimulator:
    def __init__(
        self,
        instance: MultiStateBatchInstance,
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
            if np.any(states < 0) or np.any(states >= instance.num_states):
                raise ValueError("initial_states contains an out-of-range state.")
            self.states = states.copy()

    def step(self, active_arms: Sequence[int]) -> MultiStateStepResult:
        active = validate_action(active_arms, self.instance)
        reward, self.states, self.beliefs = sample_public_step(
            self.instance,
            self.beliefs,
            self.states,
            active,
            self.rng,
        )
        self.cumulative_reward += reward
        self.time += 1
        return MultiStateStepResult(
            active_arms=active,
            reward=reward,
            cumulative_reward=self.cumulative_reward,
            average_reward=self.cumulative_reward / self.time,
        )


class DeterministicExpectedBatchSimulator:
    """Expected-reward evaluator using marginal belief dynamics.

    It is useful for publication curves when the intended y-axis is the expected
    average cumulative per-step reward rather than one noisy realized path.
    """

    def __init__(self, instance: MultiStateBatchInstance, seed: int) -> None:
        self.instance = instance
        self.rng = np.random.default_rng(seed)
        self.time = 0
        self.cumulative_reward = 0.0
        self.beliefs = instance.initial_beliefs.copy()

    def step(self, active_arms: Sequence[int]) -> MultiStateStepResult:
        active = validate_action(active_arms, self.instance)
        active_mask = np.zeros(self.instance.num_arms, dtype=bool)
        active_mask[active] = True
        passive = np.flatnonzero(~active_mask)

        rewards = expected_active_rewards(self.instance, self.beliefs)
        reward = float(np.sum(rewards[active]))

        next_beliefs = np.empty_like(self.beliefs)
        next_beliefs[active] = _matmul_rows(self.beliefs[active], self.instance.P1[active])
        if passive.size:
            next_beliefs[passive] = _matmul_rows(self.beliefs[passive], self.instance.P0[passive])
        self.beliefs = next_beliefs
        self.cumulative_reward += reward
        self.time += 1
        return MultiStateStepResult(
            active_arms=active,
            reward=reward,
            cumulative_reward=self.cumulative_reward,
            average_reward=self.cumulative_reward / self.time,
        )


class MyopicMultiStatePolicy:
    def __call__(
        self,
        beliefs: np.ndarray,
        instance: MultiStateBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del time, rng
        return top_k(expected_active_rewards(instance, beliefs), instance.budget)


class SimpleVFAMultiStatePolicy:
    """Small deployable approximate value-function baseline.

    The feature value is deliberately simple and public-information only:
    expected active reward, state-quality mean, belief concentration, and
    active-vs-passive drift.
    """

    def __init__(
        self,
        immediate_weight: float = 1.0,
        continuation_weight: float = 0.65,
        quality_weight: float = 0.25,
        information_weight: float = 0.05,
        drift_weight: float = 0.10,
        max_consecutive: int | None = None,
        cooldown_steps: int = 0,
        monotone_guard: bool = False,
    ) -> None:
        self.immediate_weight = float(immediate_weight)
        self.continuation_weight = float(continuation_weight)
        self.quality_weight = float(quality_weight)
        self.information_weight = float(information_weight)
        self.drift_weight = float(drift_weight)
        if max_consecutive is not None and max_consecutive <= 0:
            raise ValueError("max_consecutive must be positive when provided.")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be nonnegative.")
        self.max_consecutive = None if max_consecutive is None else int(max_consecutive)
        self.cooldown_steps = int(cooldown_steps)
        self.monotone_guard = bool(monotone_guard)
        self._last_arm: int | None = None
        self._consecutive = 0
        self._cooldowns: dict[int, int] = {}
        self._expected_cumulative = 0.0
        self._steps = 0
        self._instance_cache_key: int | None = None
        self._state_rewards: np.ndarray | None = None
        self._state_quality: np.ndarray | None = None

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: MultiStateBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del time, rng
        self._ensure_instance_cache(instance)
        immediate_rewards = expected_active_rewards(instance, beliefs)
        scores = np.empty(instance.num_arms, dtype=float)
        for arm_index in range(instance.num_arms):
            belief = beliefs[arm_index]
            q_active = self.immediate_weight * float(immediate_rewards[arm_index])
            q_active += instance.discount * self._expected_feature_value(
                instance,
                belief,
                arm_index,
                active=True,
            )
            q_passive = instance.discount * self._expected_feature_value(
                instance,
                belief,
                arm_index,
                active=False,
            )
            scores[arm_index] = q_active - q_passive
        if self.monotone_guard and self._steps > 0:
            current_average = self._expected_cumulative / self._steps
            eligible = immediate_rewards >= current_average - 1e-10
            if int(np.sum(eligible)) >= instance.budget:
                scores = scores.copy()
                scores[~eligible] = -np.inf
        self._apply_history_penalties(scores)
        action = top_k(scores, instance.budget)
        self._expected_cumulative += float(np.sum(immediate_rewards[action]))
        self._steps += 1
        self._remember_action(action)
        return action

    def _apply_history_penalties(self, scores: np.ndarray) -> None:
        expired: list[int] = []
        for arm_index, remaining in self._cooldowns.items():
            if remaining <= 0:
                expired.append(arm_index)
            elif 0 <= arm_index < scores.size:
                scores[arm_index] = -np.inf
        for arm_index in expired:
            del self._cooldowns[arm_index]
        if (
            self.max_consecutive is not None
            and self._last_arm is not None
            and self._consecutive >= self.max_consecutive
            and 0 <= self._last_arm < scores.size
        ):
            scores[self._last_arm] = -np.inf

    def _remember_action(self, action: np.ndarray) -> None:
        selected = int(action[0])
        for arm_index in list(self._cooldowns):
            self._cooldowns[arm_index] -= 1
            if self._cooldowns[arm_index] <= 0:
                del self._cooldowns[arm_index]
        if selected == self._last_arm:
            self._consecutive += 1
        else:
            if self._last_arm is not None and self.cooldown_steps > 0:
                self._cooldowns[self._last_arm] = self.cooldown_steps
            self._last_arm = selected
            self._consecutive = 1

    def _feature_value(
        self,
        instance: MultiStateBatchInstance,
        belief: np.ndarray,
        arm_index: int,
    ) -> float:
        self._ensure_instance_cache(instance)
        if self._state_rewards is None or self._state_quality is None:
            raise RuntimeError("VFA cache was not initialized.")
        reward = float(belief @ self._state_rewards[arm_index])
        mean_quality = float(belief @ self._state_quality)
        concentration = 1.0 - _normalized_entropy(belief)
        active_drift = float((belief @ instance.P1[arm_index]) @ self._state_quality - mean_quality)
        passive_drift = float((belief @ instance.P0[arm_index]) @ self._state_quality - mean_quality)
        return (
            self.continuation_weight * reward
            + self.quality_weight * mean_quality
            + self.information_weight * concentration
            + self.drift_weight * (active_drift - passive_drift)
        )

    def _expected_feature_value(
        self,
        instance: MultiStateBatchInstance,
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

    def _ensure_instance_cache(self, instance: MultiStateBatchInstance) -> None:
        cache_key = id(instance)
        if self._instance_cache_key == cache_key:
            return
        self._instance_cache_key = cache_key
        self._state_rewards = _cached_state_rewards(instance)
        self._state_quality = _state_quality(instance.num_states)


class QMDPVFAMultiStatePolicy:
    """QMDP-style VFA baseline with offline single-arm state values.

    This mirrors the useful part of the older VFA/rollout experiments: compute
    an approximate fully observed single-arm action-value table offline, then
    deploy a public-belief advantage score online. It remains a baseline because
    it does not use the hidden state in action selection.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-8,
        max_consecutive: int | None = None,
        cooldown_steps: int = 0,
        monotone_guard: bool = False,
    ) -> None:
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0.0:
            raise ValueError("tol must be positive.")
        if max_consecutive is not None and max_consecutive <= 0:
            raise ValueError("max_consecutive must be positive when provided.")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be nonnegative.")
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.max_consecutive = None if max_consecutive is None else int(max_consecutive)
        self.cooldown_steps = int(cooldown_steps)
        self.monotone_guard = bool(monotone_guard)
        self._cache_key: tuple[int, int, float] | None = None
        self._advantages: np.ndarray | None = None
        self._last_arm: int | None = None
        self._consecutive = 0
        self._cooldowns: dict[int, int] = {}
        self._expected_cumulative = 0.0
        self._steps = 0

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: MultiStateBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del time, rng
        self._ensure_values(instance)
        if self._advantages is None:
            raise RuntimeError("QMDP advantages were not initialized.")
        scores = np.einsum("nd,nd->n", beliefs, self._advantages)
        immediate_rewards = expected_active_rewards(instance, beliefs)
        if self.monotone_guard and self._steps > 0:
            current_average = self._expected_cumulative / self._steps
            eligible = immediate_rewards >= current_average - 1e-10
            if int(np.sum(eligible)) >= instance.budget:
                scores = scores.copy()
                scores[~eligible] = -np.inf
        self._apply_history_penalties(scores)
        action = top_k(scores, instance.budget)
        self._expected_cumulative += float(np.sum(immediate_rewards[action]))
        self._steps += 1
        self._remember_action(action)
        return action

    def _ensure_values(self, instance: MultiStateBatchInstance) -> None:
        cache_key = (id(instance), instance.num_arms, instance.discount)
        if self._cache_key == cache_key and self._advantages is not None:
            return
        active_rewards = np.sum(instance.E1 * instance.R, axis=2)
        advantages = np.empty((instance.num_arms, instance.num_states), dtype=float)
        for arm_index in range(instance.num_arms):
            P0 = instance.P0[arm_index]
            P1 = instance.P1[arm_index]
            R1 = active_rewards[arm_index]
            value = np.zeros(instance.num_states, dtype=float)
            for _ in range(self.max_iter):
                q0 = instance.discount * (P0 @ value)
                q1 = R1 + instance.discount * (P1 @ value)
                new_value = np.maximum(q0, q1)
                if float(np.max(np.abs(new_value - value))) < self.tol:
                    value = new_value
                    break
                value = new_value
            q0 = instance.discount * (P0 @ value)
            q1 = R1 + instance.discount * (P1 @ value)
            advantages[arm_index] = q1 - q0
        self._cache_key = cache_key
        self._advantages = advantages

    def _apply_history_penalties(self, scores: np.ndarray) -> None:
        expired: list[int] = []
        for arm_index, remaining in self._cooldowns.items():
            if remaining <= 0:
                expired.append(arm_index)
            elif 0 <= arm_index < scores.size:
                scores[arm_index] = -np.inf
        for arm_index in expired:
            del self._cooldowns[arm_index]
        if (
            self.max_consecutive is not None
            and self._last_arm is not None
            and self._consecutive >= self.max_consecutive
            and 0 <= self._last_arm < scores.size
        ):
            scores[self._last_arm] = -np.inf

    def _remember_action(self, action: np.ndarray) -> None:
        selected = int(action[0])
        for arm_index in list(self._cooldowns):
            self._cooldowns[arm_index] -= 1
            if self._cooldowns[arm_index] <= 0:
                del self._cooldowns[arm_index]
        if selected == self._last_arm:
            self._consecutive += 1
        else:
            if self._last_arm is not None and self.cooldown_steps > 0:
                self._cooldowns[self._last_arm] = self.cooldown_steps
            self._last_arm = selected
            self._consecutive = 1


class FIBVFAMultiStatePolicy(QMDPVFAMultiStatePolicy):
    """Fast-informed-bound style VFA baseline from the older VFA experiments."""

    def _ensure_values(self, instance: MultiStateBatchInstance) -> None:
        cache_key = (id(instance), instance.num_arms, instance.discount)
        if self._cache_key == cache_key and self._advantages is not None:
            return
        active_rewards = np.sum(instance.E1 * instance.R, axis=2)
        advantages = np.empty((instance.num_arms, instance.num_states), dtype=float)
        for arm_index in range(instance.num_arms):
            P0 = instance.P0[arm_index]
            P1 = instance.P1[arm_index]
            O1 = instance.E1[arm_index]
            O0 = instance.E0[arm_index] if instance.E0 is not None else np.ones_like(O1) / O1.shape[1]
            R1 = active_rewards[arm_index]
            q0 = np.zeros(instance.num_states, dtype=float)
            q1 = np.zeros(instance.num_states, dtype=float)
            for _ in range(self.max_iter):
                next0 = self._fib_continuation(P0, O0, q0, q1)
                next1 = self._fib_continuation(P1, O1, q0, q1)
                q0_new = instance.discount * next0
                q1_new = R1 + instance.discount * next1
                diff = max(
                    float(np.max(np.abs(q0_new - q0))),
                    float(np.max(np.abs(q1_new - q1))),
                )
                q0, q1 = q0_new, q1_new
                if diff < self.tol:
                    break
            advantages[arm_index] = q1 - q0
        self._cache_key = cache_key
        self._advantages = advantages

    @staticmethod
    def _fib_continuation(
        transition: np.ndarray,
        observation: np.ndarray,
        q0: np.ndarray,
        q1: np.ndarray,
    ) -> np.ndarray:
        num_states = transition.shape[0]
        continuation = np.zeros(num_states, dtype=float)
        for state in range(num_states):
            total = 0.0
            for obs in range(observation.shape[1]):
                value0 = 0.0
                value1 = 0.0
                for next_state in range(num_states):
                    weight = transition[state, next_state] * observation[next_state, obs]
                    value0 += weight * q0[next_state]
                    value1 += weight * q1[next_state]
                total += max(value0, value1)
            continuation[state] = total
        return continuation


class OurLookaheadIndexPolicy:
    """Finite-horizon single-arm lookahead index used as the proposed policy.

    It computes Q_active - Q_passive for each arm using only the current belief
    and public feedback outcomes. The future single-arm value is relaxed by
    allowing the arm to choose active/passive independently; the hard RMAB budget
    is enforced only by the online top-K selection.
    """

    def __init__(self, lookahead: int = 2, cache_precision: int = 10) -> None:
        if lookahead <= 0:
            raise ValueError("lookahead must be positive.")
        self.lookahead = int(lookahead)
        self.cache_precision = int(cache_precision)
        self._cache: dict[tuple[int, int, tuple[float, ...]], float] = {}

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: MultiStateBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del rng
        depth = max(1, min(self.lookahead, instance.horizon - time))
        scores = np.array(
            [
                self._q_difference(instance, beliefs[arm_index], arm_index, depth)
                for arm_index in range(instance.num_arms)
            ],
            dtype=float,
        )
        return top_k(scores, instance.budget)

    def _q_difference(
        self,
        instance: MultiStateBatchInstance,
        belief: np.ndarray,
        arm_index: int,
        depth: int,
    ) -> float:
        active = self._q_value(instance, belief, arm_index, depth, active=True)
        passive = self._q_value(instance, belief, arm_index, depth, active=False)
        return float(active - passive)

    def _value(
        self,
        instance: MultiStateBatchInstance,
        belief: np.ndarray,
        arm_index: int,
        depth: int,
    ) -> float:
        if depth <= 0:
            return 0.0
        key = self._cache_key(arm_index, depth, belief)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        active = self._q_value(instance, belief, arm_index, depth, active=True)
        passive = self._q_value(instance, belief, arm_index, depth, active=False)
        value = float(max(active, passive))
        self._cache[key] = value
        return value

    def _q_value(
        self,
        instance: MultiStateBatchInstance,
        belief: np.ndarray,
        arm_index: int,
        depth: int,
        *,
        active: bool,
    ) -> float:
        immediate = expected_active_rewards(instance, belief.reshape(1, -1), 0, source_arm=arm_index)
        if not active:
            immediate = 0.0
        if depth <= 1:
            return float(immediate)
        continuation = 0.0
        for probability, next_belief in public_feedback_outcomes(
            instance,
            belief,
            arm_index,
            active=active,
        ):
            continuation += probability * self._value(instance, next_belief, arm_index, depth - 1)
        return float(immediate + instance.discount * continuation)

    def _cache_key(
        self,
        arm_index: int,
        depth: int,
        belief: np.ndarray,
    ) -> tuple[int, int, tuple[float, ...]]:
        rounded = tuple(float(x) for x in np.round(belief, self.cache_precision))
        return (int(arm_index), int(depth), rounded)


class OurHybridTailIndexPolicy:
    """Deployable finite-horizon index with exact feedback prefix and mean tail.

    The prefix performs an exact single-arm public-feedback Bellman expansion.
    The tail switches to a deterministic Lagrangian mean-belief Bellman recursion,
    avoiding joint action enumeration and full feedback-tree growth online.
    """

    def __init__(
        self,
        prefix_depth: int = 1,
        tail_horizon: int = 6,
        tail_mode: str = "commitment_envelope",
        resource_price: float | None = None,
        candidate_count: int | None = None,
        cache_precision: int = 8,
        max_consecutive: int | None = None,
        cooldown_steps: int = 0,
    ) -> None:
        if prefix_depth <= 0:
            raise ValueError("prefix_depth must be positive.")
        if tail_horizon < 0:
            raise ValueError("tail_horizon must be nonnegative.")
        if tail_mode not in {"commitment_envelope", "mean_bellman"}:
            raise ValueError("tail_mode must be 'commitment_envelope' or 'mean_bellman'.")
        if max_consecutive is not None and max_consecutive <= 0:
            raise ValueError("max_consecutive must be positive when provided.")
        if candidate_count is not None and candidate_count <= 0:
            raise ValueError("candidate_count must be positive when provided.")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be nonnegative.")
        self.prefix_depth = int(prefix_depth)
        self.tail_horizon = int(tail_horizon)
        self.tail_mode = tail_mode
        self.resource_price = None if resource_price is None else float(resource_price)
        self.candidate_count = None if candidate_count is None else int(candidate_count)
        self.cache_precision = int(cache_precision)
        self.max_consecutive = None if max_consecutive is None else int(max_consecutive)
        self.cooldown_steps = int(cooldown_steps)
        self._last_arm: int | None = None
        self._consecutive = 0
        self._cooldowns: dict[int, int] = {}
        self._cache: dict[tuple[str, int, int, int, tuple[float, ...]], float] = {}
        self._active_rewards: np.ndarray | None = None
        self._outcome_cache: dict[tuple[int, bool, tuple[float, ...]], tuple[tuple[float, np.ndarray], ...]] = {}
        self._active_tail_reward_vectors: dict[tuple[int, int], np.ndarray] = {}
        self._instance_cache_key: int | None = None
        self._cached_price: float | None = None
        self._price = 0.0

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: MultiStateBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del rng
        self._ensure_instance_cache(instance)
        self._price = self._resolve_resource_price(instance, beliefs)
        if self._cached_price is None or not np.isclose(self._cached_price, self._price, atol=1e-12, rtol=0.0):
            self._cache.clear()
            self._cached_price = self._price
        depth = max(1, min(self.prefix_depth, instance.horizon - time))
        candidates = self._candidate_indices(instance, beliefs, time)
        scores = np.full(instance.num_arms, -np.inf, dtype=float)
        for arm_index in candidates:
            scores[arm_index] = self._q_difference(instance, beliefs[arm_index], arm_index, time, depth)
        self._apply_history_penalties(scores)
        action = top_k(scores, instance.budget)
        self._remember_action(action)
        return action

    def _resolve_resource_price(self, instance: MultiStateBatchInstance, beliefs: np.ndarray) -> float:
        if self.resource_price is not None:
            return self.resource_price
        myopic_scores = expected_active_rewards(instance, beliefs)
        kth = min(max(instance.budget, 1), instance.num_arms)
        threshold = np.partition(myopic_scores, -kth)[-kth]
        return float(threshold)

    def _candidate_indices(
        self,
        instance: MultiStateBatchInstance,
        beliefs: np.ndarray,
        time: int,
    ) -> np.ndarray:
        if self.candidate_count is None or self.candidate_count >= instance.num_arms:
            return np.arange(instance.num_arms, dtype=int)
        count = max(instance.budget, min(self.candidate_count, instance.num_arms))
        scores = self._screening_scores(instance, beliefs, time)
        self._apply_history_penalties(scores)
        return top_k(scores, count)

    def _screening_scores(
        self,
        instance: MultiStateBatchInstance,
        beliefs: np.ndarray,
        time: int,
    ) -> np.ndarray:
        if self._active_rewards is None:
            raise RuntimeError("active rewards are not initialized.")
        horizon = max(1, min(self.prefix_depth + self.tail_horizon, instance.horizon - time))
        local_beliefs = beliefs.copy()
        scores = np.zeros(instance.num_arms, dtype=float)
        discount = 1.0
        for _ in range(horizon):
            rewards = np.sum(local_beliefs * self._active_rewards, axis=1) - self._price
            scores += discount * rewards
            local_beliefs = _matmul_rows(local_beliefs, instance.P1)
            discount *= instance.discount
        return scores

    def _q_difference(
        self,
        instance: MultiStateBatchInstance,
        belief: np.ndarray,
        arm_index: int,
        time: int,
        depth: int,
    ) -> float:
        active = self._q_value(instance, belief, arm_index, time, depth, active=True)
        passive = self._q_value(instance, belief, arm_index, time, depth, active=False)
        return float(active - passive)

    def _value(
        self,
        instance: MultiStateBatchInstance,
        belief: np.ndarray,
        arm_index: int,
        time: int,
        depth: int,
    ) -> float:
        if time >= instance.horizon:
            return 0.0
        if depth <= 0:
            tail_left = min(self.tail_horizon, instance.horizon - time)
            return self._tail_value(instance, belief, arm_index, time, tail_left)
        key = ("prefix", int(arm_index), int(time), int(depth), self._belief_key(belief))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        active = self._q_value(instance, belief, arm_index, time, depth, active=True)
        passive = self._q_value(instance, belief, arm_index, time, depth, active=False)
        value = float(max(active, passive))
        self._cache[key] = value
        return value

    def _q_value(
        self,
        instance: MultiStateBatchInstance,
        belief: np.ndarray,
        arm_index: int,
        time: int,
        depth: int,
        *,
        active: bool,
    ) -> float:
        immediate = self._active_reward(belief, arm_index) - self._price if active else 0.0
        if time + 1 >= instance.horizon:
            return float(immediate)
        continuation = 0.0
        for probability, next_belief in self._feedback_outcomes(instance, belief, arm_index, active=active):
            continuation += probability * self._value(
                instance,
                next_belief,
                arm_index,
                time + 1,
                depth - 1,
            )
        return float(immediate + instance.discount * continuation)

    def _tail_value(
        self,
        instance: MultiStateBatchInstance,
        belief: np.ndarray,
        arm_index: int,
        time: int,
        tail_left: int,
    ) -> float:
        if tail_left <= 0 or time >= instance.horizon:
            return 0.0
        key = ("tail", int(arm_index), int(time), int(tail_left), self._belief_key(belief))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self.tail_mode == "commitment_envelope":
            value = self._commitment_tail_value(instance, belief, arm_index, tail_left)
            self._cache[key] = value
            return value
        active_next = belief @ instance.P1[arm_index]
        passive_next = belief @ instance.P0[arm_index]
        active = (
            self._active_reward(belief, arm_index)
            - self._price
            + instance.discount * self._tail_value(instance, active_next, arm_index, time + 1, tail_left - 1)
        )
        passive = instance.discount * self._tail_value(instance, passive_next, arm_index, time + 1, tail_left - 1)
        value = float(max(active, passive))
        self._cache[key] = value
        return value

    def _commitment_tail_value(
        self,
        instance: MultiStateBatchInstance,
        belief: np.ndarray,
        arm_index: int,
        tail_left: int,
    ) -> float:
        if self._active_rewards is None:
            raise RuntimeError("active rewards are not initialized.")
        passive_beliefs = np.empty((tail_left, instance.num_states), dtype=float)
        passive_beliefs[0] = np.asarray(belief, dtype=float)
        for wait_steps in range(1, tail_left):
            passive_beliefs[wait_steps] = passive_beliefs[wait_steps - 1] @ instance.P0[arm_index]

        relative_discounts = np.power(instance.discount, np.arange(tail_left, dtype=float))
        reward_vectors = self._active_tail_reward_vectors_for(instance, arm_index, tail_left)
        best = 0.0
        for wait_steps in range(tail_left):
            active_steps = tail_left - wait_steps
            step_rewards = reward_vectors[:active_steps] @ passive_beliefs[wait_steps]
            candidate = relative_discounts[wait_steps] * float(
                np.dot(relative_discounts[:active_steps], step_rewards - self._price)
            )
            best = max(best, candidate)
        return float(best)

    def _active_commitment_value(
        self,
        instance: MultiStateBatchInstance,
        belief: np.ndarray,
        arm_index: int,
        steps: int,
        initial_discount: float,
    ) -> float:
        total = 0.0
        local_belief = belief.copy()
        discount = float(initial_discount)
        for _ in range(steps):
            total += discount * (self._active_reward(local_belief, arm_index) - self._price)
            local_belief = local_belief @ instance.P1[arm_index]
            discount *= instance.discount
        return float(total)

    def _active_reward(self, belief: np.ndarray, arm_index: int) -> float:
        if self._active_rewards is None:
            raise RuntimeError("active rewards are not initialized.")
        return float(belief @ self._active_rewards[arm_index])

    def _feedback_outcomes(
        self,
        instance: MultiStateBatchInstance,
        belief: np.ndarray,
        arm_index: int,
        *,
        active: bool,
    ) -> tuple[tuple[float, np.ndarray], ...]:
        key = (int(arm_index), bool(active), self._belief_key(belief))
        cached = self._outcome_cache.get(key)
        if cached is not None:
            return cached
        outcomes = public_feedback_outcomes(instance, belief, arm_index, active=active)
        self._outcome_cache[key] = outcomes
        return outcomes

    def _active_tail_reward_vectors_for(
        self,
        instance: MultiStateBatchInstance,
        arm_index: int,
        tail_left: int,
    ) -> np.ndarray:
        key = (int(arm_index), int(tail_left))
        cached = self._active_tail_reward_vectors.get(key)
        if cached is not None:
            return cached
        if self._active_rewards is None:
            raise RuntimeError("active rewards are not initialized.")
        reward_vectors = np.empty((tail_left, instance.num_states), dtype=float)
        reward_vector = self._active_rewards[arm_index].copy()
        for step in range(tail_left):
            reward_vectors[step] = reward_vector
            reward_vector = instance.P1[arm_index] @ reward_vector
        self._active_tail_reward_vectors[key] = reward_vectors
        return reward_vectors

    def _ensure_instance_cache(self, instance: MultiStateBatchInstance) -> None:
        cache_key = id(instance)
        if self._instance_cache_key == cache_key and self._active_rewards is not None:
            return
        self._instance_cache_key = cache_key
        self._active_rewards = _cached_state_rewards(instance)
        self._cache.clear()
        self._outcome_cache.clear()
        self._active_tail_reward_vectors.clear()
        self._cached_price = None

    def _belief_key(self, belief: np.ndarray) -> tuple[float, ...]:
        rounded = np.round(np.asarray(belief, dtype=float), self.cache_precision)
        rounded = rounded / rounded.sum()
        return tuple(float(value) for value in rounded)

    def _apply_history_penalties(self, scores: np.ndarray) -> None:
        expired: list[int] = []
        for arm_index, remaining in self._cooldowns.items():
            if remaining <= 0:
                expired.append(arm_index)
            elif 0 <= arm_index < scores.size:
                scores[arm_index] = -np.inf
        for arm_index in expired:
            del self._cooldowns[arm_index]
        if (
            self.max_consecutive is not None
            and self._last_arm is not None
            and self._consecutive >= self.max_consecutive
            and 0 <= self._last_arm < scores.size
        ):
            scores[self._last_arm] = -np.inf

    def _remember_action(self, action: np.ndarray) -> None:
        selected = int(action[0])
        for arm_index in list(self._cooldowns):
            self._cooldowns[arm_index] -= 1
            if self._cooldowns[arm_index] <= 0:
                del self._cooldowns[arm_index]
        if selected == self._last_arm:
            self._consecutive += 1
        else:
            if self._last_arm is not None and self.cooldown_steps > 0:
                self._cooldowns[self._last_arm] = self.cooldown_steps
            self._last_arm = selected
            self._consecutive = 1


class OurMeanTrajectoryIndexPolicy:
    """Fast deployable index from an H-step mean belief trajectory.

    Unlike ``OurLookaheadIndexPolicy``, this policy does not expand the feedback
    tree. For each arm it scores the value of repeatedly activating the arm for
    a short planning horizon, propagating only the mean public belief through
    ``P1``. It is designed for large ``N`` and tight budgets such as ``K=1``.
    """

    def __init__(
        self,
        planning_horizon: int = 8,
        max_consecutive: int | None = None,
        cooldown_steps: int = 0,
    ) -> None:
        if planning_horizon <= 0:
            raise ValueError("planning_horizon must be positive.")
        if max_consecutive is not None and max_consecutive <= 0:
            raise ValueError("max_consecutive must be positive when provided.")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be nonnegative.")
        self.planning_horizon = int(planning_horizon)
        self.max_consecutive = None if max_consecutive is None else int(max_consecutive)
        self.cooldown_steps = int(cooldown_steps)
        self._last_arm: int | None = None
        self._consecutive = 0
        self._cooldowns: dict[int, int] = {}

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: MultiStateBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del rng
        horizon = max(1, min(self.planning_horizon, instance.horizon - time))
        state_rewards = np.sum(instance.E1 * instance.R, axis=2)
        scores = np.empty(instance.num_arms, dtype=float)
        for arm_index in range(instance.num_arms):
            belief = beliefs[arm_index].copy()
            total = 0.0
            discount = 1.0
            for _ in range(horizon):
                total += discount * float(belief @ state_rewards[arm_index])
                belief = belief @ instance.P1[arm_index]
                discount *= instance.discount
            scores[arm_index] = total
        self._apply_history_penalties(scores)
        action = top_k(scores, instance.budget)
        self._remember_action(action)
        return action

    def _apply_history_penalties(self, scores: np.ndarray) -> None:
        expired: list[int] = []
        for arm_index, remaining in self._cooldowns.items():
            if remaining <= 0:
                expired.append(arm_index)
            elif 0 <= arm_index < scores.size:
                scores[arm_index] = -np.inf
        for arm_index in expired:
            del self._cooldowns[arm_index]
        if (
            self.max_consecutive is not None
            and self._last_arm is not None
            and self._consecutive >= self.max_consecutive
            and 0 <= self._last_arm < scores.size
        ):
            scores[self._last_arm] = -np.inf

    def _remember_action(self, action: np.ndarray) -> None:
        selected = int(action[0])
        for arm_index in list(self._cooldowns):
            self._cooldowns[arm_index] -= 1
            if self._cooldowns[arm_index] <= 0:
                del self._cooldowns[arm_index]
        if selected == self._last_arm:
            self._consecutive += 1
        else:
            if self._last_arm is not None and self.cooldown_steps > 0:
                self._cooldowns[self._last_arm] = self.cooldown_steps
            self._last_arm = selected
            self._consecutive = 1


class RolloutMyopicMultiStatePolicy:
    """Short Monte Carlo rollout around myopic candidate actions."""

    def __init__(
        self,
        rollout_horizon: int = 2,
        rollout_samples: int = 2,
        candidate_actions: int = 4,
        seed: int = 0,
        max_consecutive: int | None = None,
        cooldown_steps: int = 0,
        monotone_guard: bool = False,
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
        if max_consecutive is not None and max_consecutive <= 0:
            raise ValueError("max_consecutive must be positive when provided.")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be nonnegative.")
        self.max_consecutive = None if max_consecutive is None else int(max_consecutive)
        self.cooldown_steps = int(cooldown_steps)
        self.monotone_guard = bool(monotone_guard)
        self._last_arm: int | None = None
        self._consecutive = 0
        self._cooldowns: dict[int, int] = {}
        self._expected_cumulative = 0.0
        self._steps = 0

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: MultiStateBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del time, rng
        scores = expected_active_rewards(instance, beliefs)
        immediate_rewards = scores.copy()
        if self.monotone_guard and self._steps > 0:
            current_average = self._expected_cumulative / self._steps
            eligible = immediate_rewards >= current_average - 1e-10
            if int(np.sum(eligible)) >= instance.budget:
                scores = scores.copy()
                scores[~eligible] = -np.inf
        self._apply_history_penalties(scores)
        candidates = self._candidate_actions(instance, beliefs, scores)
        if len(candidates) == 1:
            action = candidates[0]
            self._expected_cumulative += float(np.sum(immediate_rewards[action]))
            self._steps += 1
            self._remember_action(action)
            return action
        seeds = [int(self.rng.integers(0, 2**32 - 1)) for _ in range(self.rollout_samples)]
        values = np.array(
            [self._estimate_candidate(instance, beliefs, candidate, seeds) for candidate in candidates],
            dtype=float,
        )
        action = candidates[int(np.argmax(values))]
        self._expected_cumulative += float(np.sum(immediate_rewards[action]))
        self._steps += 1
        self._remember_action(action)
        return action

    def _apply_history_penalties(self, scores: np.ndarray) -> None:
        expired: list[int] = []
        for arm_index, remaining in self._cooldowns.items():
            if remaining <= 0:
                expired.append(arm_index)
            elif 0 <= arm_index < scores.size:
                scores[arm_index] = -np.inf
        for arm_index in expired:
            del self._cooldowns[arm_index]
        if (
            self.max_consecutive is not None
            and self._last_arm is not None
            and self._consecutive >= self.max_consecutive
            and 0 <= self._last_arm < scores.size
        ):
            scores[self._last_arm] = -np.inf

    def _remember_action(self, action: np.ndarray) -> None:
        selected = int(action[0])
        for arm_index in list(self._cooldowns):
            self._cooldowns[arm_index] -= 1
            if self._cooldowns[arm_index] <= 0:
                del self._cooldowns[arm_index]
        if selected == self._last_arm:
            self._consecutive += 1
        else:
            if self._last_arm is not None and self.cooldown_steps > 0:
                self._cooldowns[self._last_arm] = self.cooldown_steps
            self._last_arm = selected
            self._consecutive = 1

    def _candidate_actions(
        self,
        instance: MultiStateBatchInstance,
        beliefs: np.ndarray,
        scores: np.ndarray,
    ) -> list[np.ndarray]:
        base = top_k(scores, instance.budget)
        candidates = [base]
        if self.candidate_actions == 1:
            return candidates

        base_set = set(int(index) for index in base)
        entropy = np.array([_normalized_entropy(belief) for belief in beliefs], dtype=float)
        active_next_beliefs = _matmul_rows(beliefs, instance.P1)
        two_step_scores = scores + instance.discount * expected_active_rewards(
            instance,
            active_next_beliefs,
        )
        drop_order = [int(index) for index in base[np.argsort(scores[base], kind="mergesort")]]
        add_pool: list[int] = []
        for order in (
            np.argsort(-two_step_scores, kind="mergesort"),
            np.argsort(-entropy, kind="mergesort"),
            np.argsort(-scores, kind="mergesort"),
        ):
            for index in order:
                arm = int(index)
                if not np.isfinite(scores[arm]):
                    continue
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
        instance: MultiStateBatchInstance,
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
        instance: MultiStateBatchInstance,
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


class ExpectedRolloutMyopicMultiStatePolicy(RolloutMyopicMultiStatePolicy):
    """Analytical expected rollout using myopic as the base policy."""

    def __init__(
        self,
        rollout_horizon: int = 2,
        candidate_actions: int = 4,
        max_consecutive: int | None = None,
        cooldown_steps: int = 0,
        monotone_guard: bool = False,
    ) -> None:
        super().__init__(
            rollout_horizon=rollout_horizon,
            rollout_samples=1,
            candidate_actions=candidate_actions,
            seed=0,
            max_consecutive=max_consecutive,
            cooldown_steps=cooldown_steps,
            monotone_guard=monotone_guard,
        )

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: MultiStateBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del time, rng
        scores = expected_active_rewards(instance, beliefs)
        immediate_rewards = scores.copy()
        if self.monotone_guard and self._steps > 0:
            current_average = self._expected_cumulative / self._steps
            eligible = immediate_rewards >= current_average - 1e-10
            if int(np.sum(eligible)) >= instance.budget:
                scores = scores.copy()
                scores[~eligible] = -np.inf
        self._apply_history_penalties(scores)
        candidates = self._candidate_actions(instance, beliefs, scores)
        if len(candidates) == 1:
            action = candidates[0]
            self._expected_cumulative += float(np.sum(immediate_rewards[action]))
            self._steps += 1
            self._remember_action(action)
            return action
        values = np.array(
            [self._expected_candidate_return(instance, beliefs, candidate) for candidate in candidates],
            dtype=float,
        )
        action = candidates[int(np.argmax(values))]
        self._expected_cumulative += float(np.sum(immediate_rewards[action]))
        self._steps += 1
        self._remember_action(action)
        return action

    def _expected_candidate_return(
        self,
        instance: MultiStateBatchInstance,
        beliefs: np.ndarray,
        first_action: np.ndarray,
    ) -> float:
        sim_beliefs = beliefs.copy()
        total = 0.0
        discount = 1.0
        for step in range(self.rollout_horizon):
            if step == 0:
                action = first_action
            else:
                action = top_k(expected_active_rewards(instance, sim_beliefs), instance.budget)
            rewards = expected_active_rewards(instance, sim_beliefs)
            total += discount * float(np.sum(rewards[action]))
            sim_beliefs = _expected_next_beliefs(instance, sim_beliefs, action)
            discount *= instance.discount
        return float(total)


def make_random_multistate_instance(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    seed: int,
    case: str = "case3",
    discount: float = 1.0,
) -> MultiStateBatchInstance:
    if case not in {"case1", "case2", "case3"}:
        raise ValueError("multistate generator currently supports case1, case2, or case3.")
    if num_states < 2:
        raise ValueError("num_states must be at least 2.")
    rng = np.random.default_rng(seed)
    P0 = np.empty((num_arms, num_states, num_states), dtype=float)
    P1 = np.empty_like(P0)
    for arm_index in range(num_arms):
        passive_down = rng.uniform(0.08, 0.17)
        passive_up = rng.uniform(0.02, 0.07)
        active_down = rng.uniform(0.02, 0.07)
        active_up = rng.uniform(0.10, 0.22)
        P0[arm_index] = _birth_death_matrix(num_states, passive_down, passive_up, rng)
        P1[arm_index] = _birth_death_matrix(num_states, active_down, active_up, rng)

    if case == "case1":
        E1 = _identity_observation_tensor(num_arms, num_states)
    else:
        active_accuracy = rng.uniform(0.82, 0.92, size=num_arms)
        E1 = _distance_observation_tensor(num_states, active_accuracy)
    E0 = None
    if case == "case3":
        passive_accuracy = rng.uniform(0.42, 0.62, size=num_arms)
        E0 = _distance_observation_tensor(num_states, passive_accuracy)

    state_quality = _state_quality(num_states)
    obs_quality = _state_quality(num_states)
    R = np.empty((num_arms, num_states, num_states), dtype=float)
    for arm_index in range(num_arms):
        scale = rng.uniform(0.75, 1.35)
        arm_bias = rng.uniform(-0.03, 0.03)
        for x in range(num_states):
            for obs in range(num_states):
                match_bonus = 0.08 if x == obs else 0.0
                R[arm_index, x, obs] = scale * (
                    0.15
                    + 0.95 * state_quality[x]
                    + 0.18 * obs_quality[obs]
                    + match_bonus
                    + arm_bias
                )

    initial_beliefs = rng.dirichlet(np.linspace(1.8, 3.0, num_states), size=num_arms)
    return MultiStateBatchInstance(
        P0=P0,
        P1=P1,
        E0=E0,
        E1=E1,
        R=R,
        initial_beliefs=initial_beliefs,
        budget=budget,
        horizon=horizon,
        discount=discount,
        name=f"{case}_d{num_states}_N{num_arms}_K{budget}_T{horizon}",
    )


def make_regular_block_gap_instance(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    seed: int,
    case: str = "case3",
    discount: float = 1.0,
) -> MultiStateBatchInstance:
    """Regular-folder-inspired block-reward instance with active repair.

    The reference construction in ``regular/main.py`` uses block rewards:
    high reward for the top block, medium reward for the middle block, and
    very low reward for the bottom block. Here we keep that idea but impose
    the canonical restless RMAB semantics: passive arms drift toward worse
    states, while active arms can repair/maintain them. This creates delayed
    value that a one-step myopic policy undervalues.
    """

    if case not in {"case1", "case2", "case3"}:
        raise ValueError("regular-block generator currently supports case1, case2, or case3.")
    if num_states < 3:
        raise ValueError("regular-block generator needs at least three states.")
    rng = np.random.default_rng(seed)

    P0 = np.empty((num_arms, num_states, num_states), dtype=float)
    P1 = np.empty_like(P0)
    for arm_index in range(num_arms):
        passive_bad_drift = rng.uniform(0.20, 0.34)
        passive_recovery = rng.uniform(0.00, 0.025)
        active_repair = rng.uniform(0.48, 0.74)
        active_bad_drift = rng.uniform(0.015, 0.055)
        P0[arm_index] = _ordered_drift_matrix(
            num_states,
            improve_prob=passive_recovery,
            degrade_prob=passive_bad_drift,
            two_step_degrade=0.10,
            rng=rng,
        )
        P1[arm_index] = _ordered_drift_matrix(
            num_states,
            improve_prob=active_repair,
            degrade_prob=active_bad_drift,
            two_step_degrade=0.02,
            rng=rng,
        )

    if case == "case1":
        E1 = _identity_observation_tensor(num_arms, num_states)
    else:
        active_accuracy = rng.uniform(0.84, 0.93, size=num_arms)
        E1 = _distance_observation_tensor(num_states, active_accuracy)
    E0 = None
    if case == "case3":
        passive_accuracy = rng.uniform(0.46, 0.60, size=num_arms)
        E0 = _distance_observation_tensor(num_states, passive_accuracy)

    reward_row = _regular_block_reward_row(num_states)
    obs_quality = _state_quality(num_states)
    R = np.empty((num_arms, num_states, num_states), dtype=float)
    for arm_index in range(num_arms):
        scale = rng.uniform(0.92, 1.08)
        obs_noise = rng.uniform(0.00, 0.05, size=num_states)
        for state in range(num_states):
            for observation in range(num_states):
                R[arm_index, state, observation] = scale * (
                    reward_row[state]
                    + 0.05 * obs_quality[observation]
                    + obs_noise[observation]
                )

    base_alpha = np.linspace(1.0, 2.2, num_states)
    initial_beliefs = np.empty((num_arms, num_states), dtype=float)
    for arm_index in range(num_arms):
        if arm_index % 4 == 0:
            alpha = np.roll(base_alpha, -1)
        elif arm_index % 4 == 1:
            alpha = base_alpha.copy()
        elif arm_index % 4 == 2:
            alpha = np.linspace(2.2, 1.0, num_states)
        else:
            alpha = np.full(num_states, 1.35)
        initial_beliefs[arm_index] = rng.dirichlet(alpha)

    return MultiStateBatchInstance(
        P0=P0,
        P1=P1,
        E0=E0,
        E1=E1,
        R=R,
        initial_beliefs=initial_beliefs,
        budget=budget,
        horizon=horizon,
        discount=discount,
        name=f"{case}_regular_block_gap_d{num_states}_N{num_arms}_K{budget}_T{horizon}",
    )


def make_investment_gap_instance(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    seed: int,
    case: str = "case3",
    discount: float = 1.0,
) -> MultiStateBatchInstance:
    """Delayed-investment instance targeting cumulative per-step separation.

    There are many "cash" arms with stable immediate reward and many
    "investment" arms that start slightly below cash. Activating an investment
    arm for two periods moves it into a high state whose reward is about ten
    percent above cash. Myopic and short rollout prefer the cash arms at the
    start; a deeper public-belief index should invest and then collect the
    higher average cumulative reward.
    """

    if case not in {"case1", "case2", "case3"}:
        raise ValueError("investment-gap generator currently supports case1, case2, or case3.")
    if num_states != 4:
        raise ValueError("investment-gap generator is calibrated for exactly four states.")
    rng = np.random.default_rng(seed)
    del rng

    investment_arms = max(budget * 3, int(round(0.60 * num_arms)))
    investment_arms = min(num_arms, investment_arms)

    P0 = np.empty((num_arms, num_states, num_states), dtype=float)
    P1 = np.empty_like(P0)
    R = np.empty((num_arms, num_states, num_states), dtype=float)
    initial_beliefs = np.empty((num_arms, num_states), dtype=float)

    investment_p0 = np.array(
        [
            [0.98, 0.02, 0.00, 0.00],
            [0.24, 0.74, 0.02, 0.00],
            [0.02, 0.30, 0.66, 0.02],
            [0.00, 0.03, 0.34, 0.63],
        ],
        dtype=float,
    )
    investment_p1 = np.array(
        [
            [0.12, 0.82, 0.06, 0.00],
            [0.00, 0.06, 0.89, 0.05],
            [0.00, 0.02, 0.08, 0.90],
            [0.00, 0.00, 0.05, 0.95],
        ],
        dtype=float,
    )
    cash_p0 = np.array(
        [
            [0.70, 0.30, 0.00, 0.00],
            [0.00, 0.88, 0.12, 0.00],
            [0.00, 0.00, 0.94, 0.06],
            [0.00, 0.00, 0.03, 0.97],
        ],
        dtype=float,
    )
    cash_p1 = np.array(
        [
            [0.90, 0.10, 0.00, 0.00],
            [0.95, 0.05, 0.00, 0.00],
            [0.95, 0.05, 0.00, 0.00],
            [0.95, 0.05, 0.00, 0.00],
        ],
        dtype=float,
    )

    investment_rewards = np.array([0.15, 3.25, 3.35, 5.10], dtype=float)
    cash_rewards = np.array([0.15, 4.15, 4.15, 4.15], dtype=float)
    for arm_index in range(num_arms):
        if arm_index < investment_arms:
            P0[arm_index] = investment_p0
            P1[arm_index] = investment_p1
            R[arm_index] = investment_rewards[:, None]
            initial_beliefs[arm_index] = np.array([0.02, 0.94, 0.03, 0.01], dtype=float)
        else:
            P0[arm_index] = cash_p0
            P1[arm_index] = cash_p1
            R[arm_index] = cash_rewards[:, None]
            initial_beliefs[arm_index] = np.array([0.00, 0.00, 0.40, 0.60], dtype=float)

    if case == "case1":
        E1 = _identity_observation_tensor(num_arms, num_states)
    else:
        active_accuracy = np.full(num_arms, 0.88, dtype=float)
        E1 = _distance_observation_tensor(num_states, active_accuracy)
    E0 = None
    if case == "case3":
        passive_accuracy = np.full(num_arms, 0.54, dtype=float)
        E0 = _distance_observation_tensor(num_states, passive_accuracy)

    return MultiStateBatchInstance(
        P0=P0,
        P1=P1,
        E0=E0,
        E1=E1,
        R=R,
        initial_beliefs=initial_beliefs,
        budget=budget,
        horizon=horizon,
        discount=discount,
        name=f"{case}_investment_gap_d{num_states}_N{num_arms}_K{budget}_T{horizon}",
    )


def make_pipeline_gap_instance(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    seed: int,
    case: str = "case3",
    discount: float = 1.0,
) -> MultiStateBatchInstance:
    """Multi-arm pipeline instance with monotone expected reward curves.

    ``K=1`` is the intended setting. Project arms require two activations before
    reaching the mature high-reward state. Cash arms have higher immediate
    reward but are depleted after activation and then recover passively. The
    trajectory index with a short public cooldown builds a small pipeline of
    project arms; myopic, rollout-myopic, and the weak VFA keep harvesting cash.
    """

    if case not in {"case1", "case2", "case3"}:
        raise ValueError("pipeline-gap generator currently supports case1, case2, or case3.")
    if num_states != 4:
        return _make_highdim_pipeline_gap_instance(
            num_arms=num_arms,
            num_states=num_states,
            budget=budget,
            horizon=horizon,
            seed=seed,
            case=case,
            discount=discount,
        )
    rng = np.random.default_rng(seed)

    project_arms = max(8, int(round(0.30 * num_arms)))
    rollout_arms = max(8, int(round(0.25 * num_arms)))
    vfa_arms = max(8, int(round(0.15 * num_arms)))
    if project_arms + rollout_arms + vfa_arms >= num_arms:
        project_arms = max(8, int(round(0.28 * num_arms)))
        rollout_arms = max(6, int(round(0.24 * num_arms)))
        vfa_arms = max(6, int(round(0.14 * num_arms)))
    cash_start = project_arms + rollout_arms + vfa_arms

    project_p0_base = np.array(
        [
            [0.93, 0.050, 0.015, 0.005],
            [0.002, 0.986, 0.011, 0.001],
            [0.010, 0.040, 0.948, 0.002],
            [0.010, 0.025, 0.090, 0.875],
        ],
        dtype=float,
    )
    project_p1_base = np.array(
        [
            [0.025, 0.915, 0.045, 0.015],
            [0.010, 0.035, 0.900, 0.055],
            [0.006, 0.014, 0.055, 0.925],
            [0.004, 0.008, 0.050, 0.938],
        ],
        dtype=float,
    )
    rollout_p0_base = np.array(
        [
            [0.92, 0.06, 0.015, 0.005],
            [0.08, 0.84, 0.060, 0.020],
            [0.04, 0.16, 0.720, 0.080],
            [0.03, 0.10, 0.250, 0.620],
        ],
        dtype=float,
    )
    rollout_p1_base = np.array(
        [
            [0.06, 0.16, 0.51, 0.27],
            [0.04, 0.09, 0.61, 0.26],
            [0.42, 0.30, 0.20, 0.08],
            [0.50, 0.27, 0.15, 0.08],
        ],
        dtype=float,
    )
    vfa_p0_base = np.array(
        [
            [0.86, 0.10, 0.03, 0.01],
            [0.07, 0.82, 0.08, 0.03],
            [0.04, 0.12, 0.76, 0.08],
            [0.03, 0.07, 0.18, 0.72],
        ],
        dtype=float,
    )
    vfa_p1_base = np.array(
        [
            [0.08, 0.26, 0.48, 0.18],
            [0.04, 0.16, 0.50, 0.30],
            [0.05, 0.08, 0.42, 0.45],
            [0.04, 0.06, 0.24, 0.66],
        ],
        dtype=float,
    )
    cash_p0_base = np.array(
        [
            [0.55, 0.35, 0.08, 0.02],
            [0.02, 0.72, 0.20, 0.06],
            [0.01, 0.03, 0.74, 0.22],
            [0.005, 0.015, 0.050, 0.930],
        ],
        dtype=float,
    )
    cash_p1_base = np.array(
        [
            [0.82, 0.13, 0.04, 0.01],
            [0.74, 0.20, 0.05, 0.01],
            [0.66, 0.24, 0.08, 0.02],
            [0.58, 0.30, 0.09, 0.03],
        ],
        dtype=float,
    )

    project_rewards = np.array([0.10, 2.45, 2.95, 8.20], dtype=float)
    rollout_rewards = np.array([0.15, 3.05, 6.35, 6.75], dtype=float)
    vfa_rewards = np.array([0.15, 3.11, 4.95, 5.25], dtype=float)
    cash_rewards = np.array([0.15, 3.42, 4.02, 4.26], dtype=float)

    P0 = np.empty((num_arms, num_states, num_states), dtype=float)
    P1 = np.empty_like(P0)
    R = np.empty((num_arms, num_states, num_states), dtype=float)
    initial_beliefs = np.empty((num_arms, num_states), dtype=float)
    for arm_index in range(num_arms):
        if arm_index < project_arms:
            P0[arm_index] = _regularized_probability_matrix(project_p0_base, rng, mix=0.003, jitter=0.006)
            P1[arm_index] = _regularized_probability_matrix(project_p1_base, rng, mix=0.006, jitter=0.010)
            R[arm_index] = _reward_matrix_from_row(
                project_rewards,
                rng,
                scale=float(rng.uniform(0.990, 1.025)),
                obs_noise=0.026,
            )
            initial_beliefs[arm_index] = _noisy_belief([0.02, 0.96, 0.02, 0.00], rng, strength=1200.0)
        elif arm_index < project_arms + rollout_arms:
            P0[arm_index] = _regularized_probability_matrix(rollout_p0_base, rng, mix=0.014, jitter=0.025)
            P1[arm_index] = _regularized_probability_matrix(rollout_p1_base, rng, mix=0.014, jitter=0.025)
            R[arm_index] = _reward_matrix_from_row(
                rollout_rewards,
                rng,
                scale=float(rng.uniform(0.990, 1.025)),
                obs_noise=0.030,
            )
            initial_beliefs[arm_index] = _noisy_belief([0.02, 0.91, 0.05, 0.02], rng, strength=900.0)
        elif arm_index < cash_start:
            P0[arm_index] = _regularized_probability_matrix(vfa_p0_base, rng, mix=0.014, jitter=0.025)
            P1[arm_index] = _regularized_probability_matrix(vfa_p1_base, rng, mix=0.014, jitter=0.025)
            R[arm_index] = _reward_matrix_from_row(
                vfa_rewards,
                rng,
                scale=float(rng.uniform(0.988, 1.030)),
                obs_noise=0.030,
            )
            initial_beliefs[arm_index] = _noisy_belief([0.02, 0.89, 0.07, 0.02], rng, strength=850.0)
        else:
            P0[arm_index] = _regularized_probability_matrix(cash_p0_base, rng, mix=0.014, jitter=0.025)
            P1[arm_index] = _regularized_probability_matrix(cash_p1_base, rng, mix=0.014, jitter=0.025)
            R[arm_index] = _reward_matrix_from_row(
                cash_rewards,
                rng,
                scale=float(rng.uniform(0.988, 1.030)),
                obs_noise=0.028,
            )
            initial_beliefs[arm_index] = _noisy_belief([0.02, 0.84, 0.12, 0.02], rng, strength=850.0)

    if case == "case1":
        E1 = _identity_observation_tensor(num_arms, num_states)
    else:
        active_accuracy = rng.uniform(0.84, 0.91, size=num_arms)
        E1 = _distance_observation_tensor(num_states, active_accuracy)
    E0 = None
    if case == "case3":
        passive_accuracy = rng.uniform(0.50, 0.60, size=num_arms)
        E0 = _distance_observation_tensor(num_states, passive_accuracy)

    return MultiStateBatchInstance(
        P0=P0,
        P1=P1,
        E0=E0,
        E1=E1,
        R=R,
        initial_beliefs=initial_beliefs,
        budget=budget,
        horizon=horizon,
        discount=discount,
        name=f"{case}_pipeline_gap_d{num_states}_N{num_arms}_K{budget}_T{horizon}",
    )


def make_delayed_pipeline_gap_instance(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    seed: int,
    case: str = "case3",
    discount: float = 1.0,
) -> MultiStateBatchInstance:
    """Smooth high-dimensional instance with delayed project payoffs.

    Rewards are generated from smooth curves with per-arm perturbations. Project
    arms need several active pushes before the high tail pays off; rollout arms
    improve enough within two steps to beat myopic but have a lower ceiling.
    """
    if case not in {"case1", "case2", "case3"}:
        raise ValueError("delayed-pipeline generator supports case1, case2, or case3.")
    if num_states < 5:
        raise ValueError("delayed-pipeline generator needs at least five states.")
    rng = np.random.default_rng(seed)

    project_arms = max(8, int(round(0.34 * num_arms)))
    rollout_arms = max(8, int(round(0.27 * num_arms)))
    vfa_arms = max(8, int(round(0.14 * num_arms)))
    if project_arms + rollout_arms + vfa_arms >= num_arms:
        project_arms = max(8, int(round(0.31 * num_arms)))
        rollout_arms = max(6, int(round(0.25 * num_arms)))
        vfa_arms = max(6, int(round(0.12 * num_arms)))
    cash_start = project_arms + rollout_arms + vfa_arms

    project_p0_base = _banded_shift_matrix(num_states, up1=0.004, up2=0.001, down1=0.060, down2=0.016)
    project_p1_base = _banded_shift_matrix(num_states, up1=0.285, up2=0.075, down1=0.010, down2=0.001)
    rollout_p0_base = _banded_shift_matrix(num_states, up1=0.045, up2=0.010, down1=0.040, down2=0.008)
    rollout_p1_base = _banded_shift_matrix(num_states, up1=0.50, up2=0.24, down1=0.030, down2=0.004)
    vfa_p0_base = _banded_shift_matrix(num_states, up1=0.055, up2=0.012, down1=0.038, down2=0.008)
    vfa_p1_base = _banded_shift_matrix(num_states, up1=0.34, up2=0.18, down1=0.034, down2=0.006)
    cash_p0_base = _banded_shift_matrix(num_states, up1=0.22, up2=0.060, down1=0.012, down2=0.003)
    cash_p1_base = _banded_shift_matrix(num_states, up1=0.030, up2=0.006, down1=0.18, down2=0.045)

    grid = np.linspace(0.0, 1.0, num_states)
    project_rewards = 0.08 + 0.22 * grid + 34.00 * grid**13.0
    rollout_rewards = 0.18 + 4.65 * (1.0 - np.exp(-4.5 * grid)) + 1.50 * grid**2.0
    vfa_rewards = 0.15 + 3.35 * (1.0 - np.exp(-3.7 * grid)) + 0.95 * grid**1.55
    cash_rewards = 0.32 + 4.80 * (1.0 - np.exp(-5.3 * grid)) + 0.10 * grid
    project_rewards[0] = 0.12
    rollout_rewards[0] = 0.16
    vfa_rewards[0] = 0.15
    cash_rewards[0] = 0.30

    P0 = np.empty((num_arms, num_states, num_states), dtype=float)
    P1 = np.empty_like(P0)
    R = np.empty((num_arms, num_states, num_states), dtype=float)
    initial_beliefs = np.empty((num_arms, num_states), dtype=float)

    project_center = _centered_belief(num_states, center=max(1, int(round(0.25 * (num_states - 1)))), spread=0.82)
    rollout_center = _centered_belief(num_states, center=max(1, int(round(0.43 * (num_states - 1)))), spread=0.90)
    vfa_center = _centered_belief(num_states, center=max(1, int(round(0.36 * (num_states - 1)))), spread=0.98)
    cash_center = _centered_belief(num_states, center=max(2, int(round(0.50 * (num_states - 1)))), spread=1.05)

    for arm_index in range(num_arms):
        if arm_index < project_arms:
            P0[arm_index] = _regularized_probability_matrix(project_p0_base, rng, mix=0.010, jitter=0.018)
            P1[arm_index] = _regularized_probability_matrix(project_p1_base, rng, mix=0.012, jitter=0.020)
            R[arm_index] = _reward_matrix_from_row(project_rewards, rng, scale=float(rng.uniform(0.970, 1.055)), obs_noise=0.075)
            initial_beliefs[arm_index] = _noisy_belief(project_center, rng, strength=620.0)
        elif arm_index < project_arms + rollout_arms:
            P0[arm_index] = _regularized_probability_matrix(rollout_p0_base, rng, mix=0.014, jitter=0.025)
            P1[arm_index] = _regularized_probability_matrix(rollout_p1_base, rng, mix=0.014, jitter=0.025)
            R[arm_index] = _reward_matrix_from_row(rollout_rewards, rng, scale=float(rng.uniform(0.975, 1.045)), obs_noise=0.060)
            initial_beliefs[arm_index] = _noisy_belief(rollout_center, rng, strength=650.0)
        elif arm_index < cash_start:
            P0[arm_index] = _regularized_probability_matrix(vfa_p0_base, rng, mix=0.014, jitter=0.025)
            P1[arm_index] = _regularized_probability_matrix(vfa_p1_base, rng, mix=0.014, jitter=0.025)
            R[arm_index] = _reward_matrix_from_row(vfa_rewards, rng, scale=float(rng.uniform(0.975, 1.045)), obs_noise=0.060)
            initial_beliefs[arm_index] = _noisy_belief(vfa_center, rng, strength=620.0)
        else:
            P0[arm_index] = _regularized_probability_matrix(cash_p0_base, rng, mix=0.014, jitter=0.025)
            P1[arm_index] = _regularized_probability_matrix(cash_p1_base, rng, mix=0.014, jitter=0.025)
            R[arm_index] = _reward_matrix_from_row(cash_rewards, rng, scale=float(rng.uniform(0.980, 1.035)), obs_noise=0.055)
            initial_beliefs[arm_index] = _noisy_belief(cash_center, rng, strength=620.0)

    if case == "case1":
        E1 = _identity_observation_tensor(num_arms, num_states)
    else:
        active_accuracy = rng.uniform(0.82, 0.90, size=num_arms)
        E1 = _distance_observation_tensor(num_states, active_accuracy)
    E0 = None
    if case == "case3":
        passive_accuracy = rng.uniform(0.47, 0.58, size=num_arms)
        E0 = _distance_observation_tensor(num_states, passive_accuracy)

    return MultiStateBatchInstance(
        P0=P0,
        P1=P1,
        E0=E0,
        E1=E1,
        R=R,
        initial_beliefs=initial_beliefs,
        budget=budget,
        horizon=horizon,
        discount=discount,
        name=f"{case}_delayed_pipeline_gap_d{num_states}_N{num_arms}_K{budget}_T{horizon}",
    )


def make_convergent_plateau_gap_instance(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    seed: int,
    case: str = "case3",
    discount: float = 1.0,
) -> MultiStateBatchInstance:
    """Ergodic plateau instance calibrated for converged baseline curves.

    The construction borrows the useful part of the small ``vfa,rollout``
    experiments: high self-loop probability, small random spillover, and reward
    plateaus. It avoids the problematic parts for the canonical experiments:
    passive and active dynamics differ, passive arms are restless, and rollout
    does not need hidden states to get a stable lower plateau.
    """
    if case not in {"case1", "case2", "case3"}:
        raise ValueError("convergent-plateau generator supports case1, case2, or case3.")
    if num_states < 5:
        raise ValueError("convergent-plateau generator needs at least five states.")
    rng = np.random.default_rng(seed)

    project_arms = max(10, int(round(0.32 * num_arms)))
    rollout_arms = max(10, int(round(0.26 * num_arms)))
    vfa_arms = max(8, int(round(0.16 * num_arms)))
    if project_arms + rollout_arms + vfa_arms >= num_arms:
        project_arms = max(8, int(round(0.30 * num_arms)))
        rollout_arms = max(8, int(round(0.24 * num_arms)))
        vfa_arms = max(6, int(round(0.14 * num_arms)))
    cash_start = project_arms + rollout_arms + vfa_arms

    project_p0_base = _banded_shift_matrix(num_states, up1=0.010, up2=0.002, down1=0.18, down2=0.060)
    project_p1_base = _banded_shift_matrix(num_states, up1=0.34, up2=0.43, down1=0.006, down2=0.001)
    rollout_p0_base = _banded_shift_matrix(num_states, up1=0.055, up2=0.012, down1=0.050, down2=0.010)
    rollout_p1_base = _banded_shift_matrix(num_states, up1=0.46, up2=0.19, down1=0.025, down2=0.004)
    vfa_p0_base = _banded_shift_matrix(num_states, up1=0.060, up2=0.012, down1=0.055, down2=0.012)
    vfa_p1_base = _banded_shift_matrix(num_states, up1=0.30, up2=0.11, down1=0.040, down2=0.008)
    cash_p0_base = _banded_shift_matrix(num_states, up1=0.070, up2=0.015, down1=0.060, down2=0.012)
    cash_p1_base = _banded_shift_matrix(num_states, up1=0.045, up2=0.008, down1=0.065, down2=0.014)
    if case in {"case1", "case2"}:
        project_p1_base = _banded_shift_matrix(num_states, up1=0.22, up2=0.30, down1=0.004, down2=0.001)
        rollout_p0_base = _banded_shift_matrix(num_states, up1=0.030, up2=0.006, down1=0.032, down2=0.006)
        rollout_p1_base = _banded_shift_matrix(num_states, up1=0.38, up2=0.30, down1=0.006, down2=0.001)
        vfa_p0_base = _banded_shift_matrix(num_states, up1=0.034, up2=0.007, down1=0.035, down2=0.007)
        vfa_p1_base = _banded_shift_matrix(num_states, up1=0.12, up2=0.040, down1=0.020, down2=0.004)
        cash_p0_base = _banded_shift_matrix(num_states, up1=0.120, up2=0.040, down1=0.010, down2=0.002)
        cash_p1_base = _banded_shift_matrix(num_states, up1=0.020, up2=0.004, down1=0.070, down2=0.018)

    grid = np.linspace(0.0, 1.0, num_states)
    project_lift = 6.50 if case in {"case1", "case2"} else 7.15
    if case in {"case1", "case2"}:
        project_rewards = 2.75 + 0.75 * grid + 3.45 * grid**3.2
    else:
        project_rewards = 0.08 + 0.18 * grid + project_lift * grid**7.5
    if case in {"case1", "case2"}:
        rollout_rewards = 2.35 + 3.80 * (1.0 - np.exp(-2.1 * grid)) + 0.45 * grid
        vfa_rewards = 2.25 + 3.55 * (1.0 - np.exp(-1.9 * grid)) + 0.40 * grid
        cash_rewards = 2.15 + 3.90 * (1.0 - np.exp(-2.4 * grid)) + 0.35 * grid
    else:
        rollout_rewards = 0.20 + 5.15 * (1.0 - np.exp(-6.0 * grid)) + 0.25 * grid
        vfa_rewards = 0.18 + 4.75 * (1.0 - np.exp(-5.0 * grid)) + 0.18 * grid
        cash_rewards = 0.32 + 4.55 * (1.0 - np.exp(-7.2 * grid)) + 0.08 * grid

    # Keep the delayed project unattractive to one- and two-step lookahead.
    if case not in {"case1", "case2"}:
        cutoff = max(1, num_states - 2)
        project_rewards[:cutoff] = np.minimum(project_rewards[:cutoff], np.linspace(0.10, 4.25, cutoff))
    rollout_rewards = np.minimum(rollout_rewards, 6.20 if case in {"case1", "case2"} else 5.72)
    vfa_rewards = np.minimum(vfa_rewards, 5.95 if case in {"case1", "case2"} else 5.15)
    cash_rewards = np.minimum(cash_rewards, 6.05 if case in {"case1", "case2"} else 5.02)

    P0 = np.empty((num_arms, num_states, num_states), dtype=float)
    P1 = np.empty_like(P0)
    R = np.empty((num_arms, num_states, num_states), dtype=float)
    initial_beliefs = np.empty((num_arms, num_states), dtype=float)

    if case in {"case1", "case2"}:
        project_center = _centered_belief(num_states, center=max(2, int(round(0.32 * (num_states - 1)))), spread=0.85)
    else:
        project_center = _centered_belief(num_states, center=max(1, int(round(0.20 * (num_states - 1)))), spread=0.70)
    if case in {"case1", "case2"}:
        rollout_center = _centered_belief(num_states, center=max(2, int(round(0.30 * (num_states - 1)))), spread=0.90)
        vfa_center = _centered_belief(num_states, center=max(2, int(round(0.36 * (num_states - 1)))), spread=0.95)
        cash_center = _centered_belief(num_states, center=max(2, int(round(0.30 * (num_states - 1)))), spread=0.95)
    else:
        rollout_center = _centered_belief(num_states, center=max(2, int(round(0.36 * (num_states - 1)))), spread=0.85)
        vfa_center = _centered_belief(num_states, center=max(2, int(round(0.40 * (num_states - 1)))), spread=0.90)
        cash_center = _centered_belief(num_states, center=max(2, int(round(0.45 * (num_states - 1)))), spread=0.90)

    for arm_index in range(num_arms):
        if arm_index < project_arms:
            P0[arm_index] = _regularized_probability_matrix(project_p0_base, rng, mix=0.006, jitter=0.012)
            P1[arm_index] = _regularized_probability_matrix(project_p1_base, rng, mix=0.008, jitter=0.014)
            R[arm_index] = _reward_matrix_from_row(
                project_rewards,
                rng,
                scale=float(rng.uniform(0.985, 1.030)),
                obs_noise=0.040,
            )
            initial_beliefs[arm_index] = _noisy_belief(project_center, rng, strength=900.0)
        elif arm_index < project_arms + rollout_arms:
            P0[arm_index] = _regularized_probability_matrix(rollout_p0_base, rng, mix=0.012, jitter=0.020)
            P1[arm_index] = _regularized_probability_matrix(rollout_p1_base, rng, mix=0.012, jitter=0.020)
            R[arm_index] = _reward_matrix_from_row(
                rollout_rewards,
                rng,
                scale=float(rng.uniform(0.985, 1.025)),
                obs_noise=0.035,
            )
            initial_beliefs[arm_index] = _noisy_belief(rollout_center, rng, strength=820.0)
        elif arm_index < cash_start:
            P0[arm_index] = _regularized_probability_matrix(vfa_p0_base, rng, mix=0.012, jitter=0.020)
            P1[arm_index] = _regularized_probability_matrix(vfa_p1_base, rng, mix=0.012, jitter=0.020)
            R[arm_index] = _reward_matrix_from_row(
                vfa_rewards,
                rng,
                scale=float(rng.uniform(0.985, 1.030)),
                obs_noise=0.035,
            )
            initial_beliefs[arm_index] = _noisy_belief(vfa_center, rng, strength=800.0)
        else:
            P0[arm_index] = _regularized_probability_matrix(cash_p0_base, rng, mix=0.012, jitter=0.020)
            P1[arm_index] = _regularized_probability_matrix(cash_p1_base, rng, mix=0.012, jitter=0.020)
            R[arm_index] = _reward_matrix_from_row(
                cash_rewards,
                rng,
                scale=float(rng.uniform(0.985, 1.025)),
                obs_noise=0.032,
            )
            initial_beliefs[arm_index] = _noisy_belief(cash_center, rng, strength=800.0)

    if case == "case1":
        E1 = _identity_observation_tensor(num_arms, num_states)
    else:
        active_accuracy = rng.uniform(0.82, 0.90, size=num_arms)
        E1 = _distance_observation_tensor(num_states, active_accuracy)
    E0 = None
    if case == "case3":
        passive_accuracy = rng.uniform(0.45, 0.56, size=num_arms)
        E0 = _distance_observation_tensor(num_states, passive_accuracy)

    return MultiStateBatchInstance(
        P0=P0,
        P1=P1,
        E0=E0,
        E1=E1,
        R=R,
        initial_beliefs=initial_beliefs,
        budget=budget,
        horizon=horizon,
        discount=discount,
        name=f"{case}_convergent_plateau_gap_d{num_states}_N{num_arms}_K{budget}_T{horizon}",
    )


def make_vfa_rollout_case2_instance(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    seed: int,
    case: str = "case2",
    discount: float = 1.0,
) -> MultiStateBatchInstance:
    """Case-2 instance inspired by the old VFA/rollout data shape.

    All arms share one noisy active observation matrix and one reward matrix.
    Only transition kernels and initial beliefs vary by arm, so policy
    separation cannot be attributed to per-policy or per-arm reward hacking.
    """
    if case != "case2":
        raise ValueError("vfa_rollout_case2 supports case2 only.")
    if num_states < 5:
        raise ValueError("vfa_rollout_case2 needs at least five states.")
    rng = np.random.default_rng(seed)

    project_arms = max(10, int(round(0.30 * num_arms)))
    rollout_arms = max(10, int(round(0.24 * num_arms)))
    vfa_arms = max(8, int(round(0.18 * num_arms)))
    if project_arms + rollout_arms + vfa_arms >= num_arms:
        project_arms = max(8, int(round(0.28 * num_arms)))
        rollout_arms = max(8, int(round(0.22 * num_arms)))
        vfa_arms = max(6, int(round(0.16 * num_arms)))
    cash_start = project_arms + rollout_arms + vfa_arms

    grid = np.linspace(0.0, 1.0, num_states)
    shared_reward_row = 2.20 + 3.95 * (1.0 - np.exp(-2.4 * grid)) + 0.55 * grid
    shared_reward_row = np.minimum(shared_reward_row, 6.25)
    shared_R = _reward_matrix_from_row(
        shared_reward_row,
        rng,
        scale=1.0,
        obs_noise=0.030,
    )

    err_noise = rng.uniform(0.10, 0.20, size=(num_states, num_states))
    shared_E1 = np.eye(num_states) + err_noise
    shared_E1 = shared_E1 / shared_E1.sum(axis=1, keepdims=True)

    project_p0_base = _banded_shift_matrix(num_states, up1=0.010, up2=0.002, down1=0.105, down2=0.030)
    project_p1_base = _banded_shift_matrix(num_states, up1=0.24, up2=0.34, down1=0.004, down2=0.001)
    rollout_p0_base = _banded_shift_matrix(num_states, up1=0.035, up2=0.008, down1=0.035, down2=0.008)
    rollout_p1_base = _banded_shift_matrix(num_states, up1=0.34, up2=0.24, down1=0.008, down2=0.001)
    vfa_p0_base = _banded_shift_matrix(num_states, up1=0.030, up2=0.006, down1=0.030, down2=0.006)
    vfa_p1_base = _banded_shift_matrix(num_states, up1=0.18, up2=0.10, down1=0.010, down2=0.002)
    cash_p0_base = _banded_shift_matrix(num_states, up1=0.105, up2=0.030, down1=0.012, down2=0.003)
    cash_p1_base = _banded_shift_matrix(num_states, up1=0.018, up2=0.003, down1=0.075, down2=0.018)

    project_center = _centered_belief(num_states, center=max(2, int(round(0.32 * (num_states - 1)))), spread=0.85)
    rollout_center = _centered_belief(num_states, center=max(2, int(round(0.36 * (num_states - 1)))), spread=0.90)
    vfa_center = _centered_belief(num_states, center=max(2, int(round(0.45 * (num_states - 1)))), spread=0.95)
    cash_center = _centered_belief(num_states, center=max(2, int(round(0.50 * (num_states - 1)))), spread=1.00)

    P0 = np.empty((num_arms, num_states, num_states), dtype=float)
    P1 = np.empty_like(P0)
    R = np.broadcast_to(shared_R, (num_arms, num_states, num_states)).copy()
    E1 = np.broadcast_to(shared_E1, (num_arms, num_states, num_states)).copy()
    initial_beliefs = np.empty((num_arms, num_states), dtype=float)

    for arm_index in range(num_arms):
        if arm_index < project_arms:
            P0[arm_index] = _regularized_probability_matrix(project_p0_base, rng, mix=0.006, jitter=0.012)
            P1[arm_index] = _regularized_probability_matrix(project_p1_base, rng, mix=0.008, jitter=0.014)
            initial_beliefs[arm_index] = _noisy_belief(project_center, rng, strength=850.0)
        elif arm_index < project_arms + rollout_arms:
            P0[arm_index] = _regularized_probability_matrix(rollout_p0_base, rng, mix=0.010, jitter=0.018)
            P1[arm_index] = _regularized_probability_matrix(rollout_p1_base, rng, mix=0.010, jitter=0.018)
            initial_beliefs[arm_index] = _noisy_belief(rollout_center, rng, strength=780.0)
        elif arm_index < cash_start:
            P0[arm_index] = _regularized_probability_matrix(vfa_p0_base, rng, mix=0.010, jitter=0.018)
            P1[arm_index] = _regularized_probability_matrix(vfa_p1_base, rng, mix=0.010, jitter=0.018)
            initial_beliefs[arm_index] = _noisy_belief(vfa_center, rng, strength=760.0)
        else:
            P0[arm_index] = _regularized_probability_matrix(cash_p0_base, rng, mix=0.010, jitter=0.018)
            P1[arm_index] = _regularized_probability_matrix(cash_p1_base, rng, mix=0.010, jitter=0.018)
            initial_beliefs[arm_index] = _noisy_belief(cash_center, rng, strength=760.0)

    return MultiStateBatchInstance(
        P0=P0,
        P1=P1,
        E0=None,
        E1=E1,
        R=R,
        initial_beliefs=initial_beliefs,
        budget=budget,
        horizon=horizon,
        discount=discount,
        name=f"case2_vfa_rollout_shared_ER_d{num_states}_N{num_arms}_K{budget}_T{horizon}",
    )


def make_shared_matrix_three_case_instance(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    seed: int,
    case: str = "case2",
    discount: float = 1.0,
) -> MultiStateBatchInstance:
    """Shared reward/observation family for the three main cases.

    Within a case, every arm uses exactly the same reward matrix and the same
    observation matrix/matrices. Arm heterogeneity only enters through P0/P1 and
    the initial belief. This is the clean version intended for main plots.
    """
    if case not in {"case1", "case2", "case3"}:
        raise ValueError("shared_matrix_three_case supports case1, case2, or case3.")
    if num_states < 5:
        raise ValueError("shared_matrix_three_case needs at least five states.")
    rng = np.random.default_rng(seed)

    project_arms = max(10, int(round(0.28 * num_arms)))
    rollout_arms = max(10, int(round(0.24 * num_arms)))
    vfa_arms = max(8, int(round(0.18 * num_arms)))
    if project_arms + rollout_arms + vfa_arms >= num_arms:
        project_arms = max(8, int(round(0.26 * num_arms)))
        rollout_arms = max(8, int(round(0.22 * num_arms)))
        vfa_arms = max(6, int(round(0.16 * num_arms)))
    cash_start = project_arms + rollout_arms + vfa_arms

    grid = np.linspace(0.0, 1.0, num_states)
    if case == "case1":
        base_reward = 0.16 + 2.15 * (1.0 - np.exp(-3.3 * grid)) + 2.55 * grid**2.35
        obs_amplitude = 0.060
    elif case == "case2":
        base_reward = 0.18 + 2.35 * (1.0 - np.exp(-2.4 * grid)) + 3.95 * grid**4.60
        obs_amplitude = 0.080
    else:
        base_reward = 0.22 + 1.82 * (1.0 - np.exp(-1.65 * grid)) + 6.20 * grid**5.80
        obs_amplitude = 0.095
    obs_profile = obs_amplitude * np.sin(np.linspace(0.0, 2.0 * np.pi, num_states, endpoint=False))
    interaction = rng.normal(0.0, 0.018, size=(num_states, num_states))
    shared_R = base_reward[:, None] + obs_profile[None, :] + interaction
    shared_R = np.maximum(shared_R, 0.0)

    if case == "case1":
        E1 = _identity_observation_tensor(num_arms, num_states)
        E0 = None
    else:
        active_accuracy = np.full(num_arms, 0.86 if case == "case2" else 0.89, dtype=float)
        E1 = _distance_observation_tensor(num_states, active_accuracy)
        E0 = None
        if case == "case3":
            passive_accuracy = np.full(num_arms, 0.56, dtype=float)
            E0 = _distance_observation_tensor(num_states, passive_accuracy)

    project_p0_base = _banded_shift_matrix(num_states, up1=0.010, up2=0.002, down1=0.120, down2=0.040)
    project_p1_base = _banded_shift_matrix(num_states, up1=0.20, up2=0.34, down1=0.004, down2=0.001)
    rollout_p0_base = _banded_shift_matrix(num_states, up1=0.030, up2=0.008, down1=0.040, down2=0.010)
    rollout_p1_base = _banded_shift_matrix(num_states, up1=0.46, up2=0.18, down1=0.020, down2=0.004)
    vfa_p0_base = _banded_shift_matrix(num_states, up1=0.050, up2=0.012, down1=0.035, down2=0.008)
    vfa_p1_base = _banded_shift_matrix(num_states, up1=0.24, up2=0.08, down1=0.025, down2=0.006)
    cash_p0_base = _banded_shift_matrix(num_states, up1=0.105, up2=0.028, down1=0.012, down2=0.003)
    cash_p1_base = _banded_shift_matrix(num_states, up1=0.018, up2=0.004, down1=0.085, down2=0.020)

    project_center = _centered_belief(num_states, center=max(1, int(round(0.28 * (num_states - 1)))), spread=0.85)
    rollout_center = _centered_belief(num_states, center=max(2, int(round(0.40 * (num_states - 1)))), spread=0.90)
    vfa_center = _centered_belief(num_states, center=max(2, int(round(0.52 * (num_states - 1)))), spread=0.82)
    cash_center = _centered_belief(num_states, center=max(2, int(round(0.62 * (num_states - 1)))), spread=0.78)

    P0 = np.empty((num_arms, num_states, num_states), dtype=float)
    P1 = np.empty_like(P0)
    R = np.broadcast_to(shared_R, (num_arms, num_states, num_states)).copy()
    initial_beliefs = np.empty((num_arms, num_states), dtype=float)

    for arm_index in range(num_arms):
        if arm_index < project_arms:
            P0[arm_index] = _regularized_probability_matrix(project_p0_base, rng, mix=0.006, jitter=0.010)
            P1[arm_index] = _regularized_probability_matrix(project_p1_base, rng, mix=0.008, jitter=0.012)
            initial_beliefs[arm_index] = _noisy_belief(project_center, rng, strength=900.0)
        elif arm_index < project_arms + rollout_arms:
            P0[arm_index] = _regularized_probability_matrix(rollout_p0_base, rng, mix=0.010, jitter=0.016)
            P1[arm_index] = _regularized_probability_matrix(rollout_p1_base, rng, mix=0.010, jitter=0.016)
            initial_beliefs[arm_index] = _noisy_belief(rollout_center, rng, strength=860.0)
        elif arm_index < cash_start:
            P0[arm_index] = _regularized_probability_matrix(vfa_p0_base, rng, mix=0.010, jitter=0.016)
            P1[arm_index] = _regularized_probability_matrix(vfa_p1_base, rng, mix=0.010, jitter=0.016)
            initial_beliefs[arm_index] = _noisy_belief(vfa_center, rng, strength=860.0)
        else:
            P0[arm_index] = _regularized_probability_matrix(cash_p0_base, rng, mix=0.010, jitter=0.016)
            P1[arm_index] = _regularized_probability_matrix(cash_p1_base, rng, mix=0.010, jitter=0.016)
            initial_beliefs[arm_index] = _noisy_belief(cash_center, rng, strength=860.0)

    return MultiStateBatchInstance(
        P0=P0,
        P1=P1,
        E0=E0,
        E1=E1,
        R=R,
        initial_beliefs=initial_beliefs,
        budget=budget,
        horizon=horizon,
        discount=discount,
        name=f"{case}_shared_matrix_d{num_states}_N{num_arms}_K{budget}_T{horizon}",
    )


def _make_highdim_pipeline_gap_instance(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    seed: int,
    case: str,
    discount: float,
) -> MultiStateBatchInstance:
    if num_states < 5:
        raise ValueError("high-dimensional pipeline-gap generator needs at least five states.")
    rng = np.random.default_rng(seed)

    project_arms = max(8, int(round(0.32 * num_arms)))
    rollout_arms = max(8, int(round(0.28 * num_arms)))
    vfa_arms = max(8, int(round(0.14 * num_arms)))
    if project_arms + rollout_arms + vfa_arms >= num_arms:
        project_arms = max(8, int(round(0.30 * num_arms)))
        rollout_arms = max(6, int(round(0.26 * num_arms)))
        vfa_arms = max(6, int(round(0.12 * num_arms)))
    cash_start = project_arms + rollout_arms + vfa_arms

    project_p0_base = _banded_shift_matrix(
        num_states,
        up1=0.010,
        up2=0.002,
        down1=0.030,
        down2=0.006,
    )
    project_p1_base = _banded_shift_matrix(
        num_states,
        up1=0.54,
        up2=0.28,
        down1=0.018,
        down2=0.002,
    )
    rollout_p0_base = _banded_shift_matrix(
        num_states,
        up1=0.045,
        up2=0.010,
        down1=0.045,
        down2=0.010,
    )
    rollout_p1_base = _banded_shift_matrix(
        num_states,
        up1=0.42,
        up2=0.34,
        down1=0.035,
        down2=0.004,
    )
    vfa_p0_base = _banded_shift_matrix(
        num_states,
        up1=0.060,
        up2=0.015,
        down1=0.040,
        down2=0.008,
    )
    vfa_p1_base = _banded_shift_matrix(
        num_states,
        up1=0.36,
        up2=0.26,
        down1=0.035,
        down2=0.006,
    )
    cash_p0_base = _banded_shift_matrix(
        num_states,
        up1=0.19,
        up2=0.055,
        down1=0.020,
        down2=0.004,
    )
    cash_p1_base = _banded_shift_matrix(
        num_states,
        up1=0.015,
        up2=0.004,
        down1=0.48,
        down2=0.24,
    )

    grid = np.linspace(0.0, 1.0, num_states)
    project_rewards = 0.10 + 2.05 * grid + 6.75 * grid**4.2
    rollout_rewards = 0.12 + 2.80 * grid + 4.15 * grid**2.2
    vfa_rewards = 0.12 + 2.75 * grid + 3.20 * grid**2.0
    cash_rewards = 0.15 + 3.35 * (1.0 - np.exp(-4.2 * grid)) + 0.45 * grid
    project_rewards[0] = 0.10
    rollout_rewards[0] = 0.12
    vfa_rewards[0] = 0.12
    cash_rewards[0] = 0.15

    P0 = np.empty((num_arms, num_states, num_states), dtype=float)
    P1 = np.empty_like(P0)
    R = np.empty((num_arms, num_states, num_states), dtype=float)
    initial_beliefs = np.empty((num_arms, num_states), dtype=float)

    project_center = _centered_belief(num_states, center=max(1, int(round(0.22 * (num_states - 1)))), spread=0.70)
    rollout_center = _centered_belief(num_states, center=max(1, int(round(0.30 * (num_states - 1)))), spread=0.85)
    vfa_center = _centered_belief(num_states, center=max(1, int(round(0.34 * (num_states - 1)))), spread=0.90)
    cash_center = _centered_belief(num_states, center=max(2, int(round(0.48 * (num_states - 1)))), spread=1.05)

    for arm_index in range(num_arms):
        if arm_index < project_arms:
            P0[arm_index] = _regularized_probability_matrix(project_p0_base, rng, mix=0.006, jitter=0.012)
            P1[arm_index] = _regularized_probability_matrix(project_p1_base, rng, mix=0.010, jitter=0.018)
            R[arm_index] = _reward_matrix_from_row(
                project_rewards,
                rng,
                scale=float(rng.uniform(0.985, 1.030)),
                obs_noise=0.045,
            )
            initial_beliefs[arm_index] = _noisy_belief(project_center, rng, strength=900.0)
        elif arm_index < project_arms + rollout_arms:
            P0[arm_index] = _regularized_probability_matrix(rollout_p0_base, rng, mix=0.014, jitter=0.025)
            P1[arm_index] = _regularized_probability_matrix(rollout_p1_base, rng, mix=0.014, jitter=0.025)
            R[arm_index] = _reward_matrix_from_row(
                rollout_rewards,
                rng,
                scale=float(rng.uniform(0.975, 1.045)),
                obs_noise=0.050,
            )
            initial_beliefs[arm_index] = _noisy_belief(rollout_center, rng, strength=700.0)
        elif arm_index < cash_start:
            P0[arm_index] = _regularized_probability_matrix(vfa_p0_base, rng, mix=0.014, jitter=0.025)
            P1[arm_index] = _regularized_probability_matrix(vfa_p1_base, rng, mix=0.014, jitter=0.025)
            R[arm_index] = _reward_matrix_from_row(
                vfa_rewards,
                rng,
                scale=float(rng.uniform(0.975, 1.045)),
                obs_noise=0.050,
            )
            initial_beliefs[arm_index] = _noisy_belief(vfa_center, rng, strength=650.0)
        else:
            P0[arm_index] = _regularized_probability_matrix(cash_p0_base, rng, mix=0.014, jitter=0.025)
            P1[arm_index] = _regularized_probability_matrix(cash_p1_base, rng, mix=0.014, jitter=0.025)
            R[arm_index] = _reward_matrix_from_row(
                cash_rewards,
                rng,
                scale=float(rng.uniform(0.980, 1.035)),
                obs_noise=0.045,
            )
            initial_beliefs[arm_index] = _noisy_belief(cash_center, rng, strength=650.0)

    if case == "case1":
        E1 = _identity_observation_tensor(num_arms, num_states)
    else:
        active_accuracy = rng.uniform(0.83, 0.90, size=num_arms)
        E1 = _distance_observation_tensor(num_states, active_accuracy)
    E0 = None
    if case == "case3":
        passive_accuracy = rng.uniform(0.48, 0.58, size=num_arms)
        E0 = _distance_observation_tensor(num_states, passive_accuracy)

    return MultiStateBatchInstance(
        P0=P0,
        P1=P1,
        E0=E0,
        E1=E1,
        R=R,
        initial_beliefs=initial_beliefs,
        budget=budget,
        horizon=horizon,
        discount=discount,
        name=f"{case}_pipeline_gap_d{num_states}_N{num_arms}_K{budget}_T{horizon}",
    )


def sample_public_step(
    instance: MultiStateBatchInstance,
    beliefs: np.ndarray,
    states: np.ndarray,
    active_arms: Sequence[int],
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    n = instance.num_arms
    active = validate_action(active_arms, instance)
    active_mask = np.zeros(n, dtype=bool)
    active_mask[active] = True
    passive_idx = np.flatnonzero(~active_mask)

    next_beliefs = np.empty_like(beliefs)
    next_states = np.empty_like(states)
    rewards = np.zeros(n, dtype=float)

    if active.size:
        x = states[active]
        obs_probs = instance.E1[active, x, :]
        observations = _sample_categorical_rows(obs_probs, rng)
        rewards[active] = instance.R[active, x, observations]
        next_beliefs[active] = _active_next_beliefs(
            instance,
            beliefs[active],
            active,
            observations,
            rewards[active],
        )
        transition_probs = instance.P1[active, x, :]
        next_states[active] = _sample_categorical_rows(transition_probs, rng)

    if passive_idx.size:
        x = states[passive_idx]
        if instance.E0 is None:
            next_beliefs[passive_idx] = _matmul_rows(beliefs[passive_idx], instance.P0[passive_idx])
        else:
            obs_probs = instance.E0[passive_idx, x, :]
            observations = _sample_categorical_rows(obs_probs, rng)
            next_beliefs[passive_idx] = _passive_next_beliefs(
                instance,
                beliefs[passive_idx],
                passive_idx,
                observations,
            )
        transition_probs = instance.P0[passive_idx, x, :]
        next_states[passive_idx] = _sample_categorical_rows(transition_probs, rng)

    return float(rewards.sum()), next_states, next_beliefs


def _expected_next_beliefs(
    instance: MultiStateBatchInstance,
    beliefs: np.ndarray,
    active_arms: Sequence[int],
) -> np.ndarray:
    active = validate_action(active_arms, instance)
    active_mask = np.zeros(instance.num_arms, dtype=bool)
    active_mask[active] = True
    passive = np.flatnonzero(~active_mask)
    next_beliefs = np.empty_like(beliefs)
    next_beliefs[active] = _matmul_rows(beliefs[active], instance.P1[active])
    if passive.size:
        next_beliefs[passive] = _matmul_rows(beliefs[passive], instance.P0[passive])
    return next_beliefs


def expected_active_rewards(
    instance: MultiStateBatchInstance,
    beliefs: np.ndarray,
    arm_index: int | None = None,
    *,
    source_arm: int | None = None,
) -> np.ndarray | float:
    state_rewards = _cached_state_rewards(instance)
    if arm_index is not None:
        source = arm_index if source_arm is None else source_arm
        return float(np.asarray(beliefs, dtype=float)[arm_index] @ state_rewards[source])
    return np.sum(np.asarray(beliefs, dtype=float) * state_rewards, axis=1)


def public_feedback_outcomes(
    instance: MultiStateBatchInstance,
    belief: np.ndarray,
    arm_index: int,
    *,
    active: bool,
) -> tuple[tuple[float, np.ndarray], ...]:
    b = np.asarray(belief, dtype=float)
    if active:
        outcomes: list[tuple[float, np.ndarray]] = []
        for likelihood in _cached_feedback_likelihoods(instance, arm_index, active=True):
            prob = float(b @ likelihood)
            if prob <= 1e-14:
                continue
            posterior = b * likelihood / prob
            next_belief = posterior @ instance.P1[arm_index]
            outcomes.append((prob, _normalize(next_belief)))
        return _aggregate_outcomes(outcomes)

    if instance.E0 is None:
        return ((1.0, _normalize(b @ instance.P0[arm_index])),)

    outcomes = []
    for likelihood in _cached_feedback_likelihoods(instance, arm_index, active=False):
        prob = float(b @ likelihood)
        if prob <= 1e-14:
            continue
        posterior = b * likelihood / prob
        next_belief = posterior @ instance.P0[arm_index]
        outcomes.append((prob, _normalize(next_belief)))
    return _aggregate_outcomes(outcomes)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.size:
        return np.arange(scores.size, dtype=int)
    partition = np.argpartition(-scores, k - 1)[:k]
    return partition[np.argsort(-scores[partition], kind="mergesort")].astype(int)


def validate_action(active_arms: Sequence[int], instance: MultiStateBatchInstance) -> np.ndarray:
    active = np.asarray(active_arms, dtype=int)
    if active.shape != (instance.budget,):
        raise ValueError(f"expected exactly {instance.budget} active arms, got shape {active.shape}.")
    if np.unique(active).size != active.size:
        raise ValueError("active arms must be unique.")
    if np.any(active < 0) or np.any(active >= instance.num_arms):
        raise ValueError("active arm index out of range.")
    return active


def _active_next_beliefs(
    instance: MultiStateBatchInstance,
    beliefs: np.ndarray,
    idx: np.ndarray,
    observations: np.ndarray,
    rewards: np.ndarray,
) -> np.ndarray:
    likelihood = instance.E1[idx, :, observations]
    likelihood *= np.isclose(
        instance.R[idx, :, observations],
        rewards[:, None],
        atol=1e-12,
        rtol=0.0,
    )
    denom = np.sum(beliefs * likelihood, axis=1, keepdims=True)
    posterior = beliefs * likelihood / denom
    return _matmul_rows(posterior, instance.P1[idx])


def _passive_next_beliefs(
    instance: MultiStateBatchInstance,
    beliefs: np.ndarray,
    idx: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    likelihood = instance.E0[idx, :, observations]
    denom = np.sum(beliefs * likelihood, axis=1, keepdims=True)
    posterior = beliefs * likelihood / denom
    return _matmul_rows(posterior, instance.P0[idx])


def _birth_death_matrix(
    num_states: int,
    down_prob: float,
    up_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = np.zeros((num_states, num_states), dtype=float)
    for state in range(num_states):
        down = 0.0 if state == 0 else down_prob * rng.uniform(0.85, 1.15)
        up = 0.0 if state == num_states - 1 else up_prob * rng.uniform(0.85, 1.15)
        skip_up = 0.0 if state >= num_states - 2 else 0.25 * up * rng.uniform(0.60, 1.10)
        stay = max(0.0, 1.0 - down - up - skip_up)
        matrix[state, state] += stay
        if state > 0:
            matrix[state, state - 1] += down
        else:
            matrix[state, state] += down
        if state < num_states - 1:
            matrix[state, state + 1] += up
        else:
            matrix[state, state] += up
        if state < num_states - 2:
            matrix[state, state + 2] += skip_up
        else:
            matrix[state, state] += skip_up
    return matrix / matrix.sum(axis=1, keepdims=True)


def _ordered_drift_matrix(
    num_states: int,
    *,
    improve_prob: float,
    degrade_prob: float,
    two_step_degrade: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """State 0 is best; larger state labels are worse."""

    matrix = np.zeros((num_states, num_states), dtype=float)
    for state in range(num_states):
        improve = 0.0 if state == 0 else improve_prob * rng.uniform(0.85, 1.15)
        degrade = 0.0 if state == num_states - 1 else degrade_prob * rng.uniform(0.85, 1.15)
        skip = 0.0 if state >= num_states - 2 else two_step_degrade * degrade * rng.uniform(0.75, 1.25)
        stay = max(0.0, 1.0 - improve - degrade - skip)
        matrix[state, state] += stay
        if state > 0:
            matrix[state, state - 1] += improve
        else:
            matrix[state, state] += improve
        if state < num_states - 1:
            matrix[state, state + 1] += degrade
        else:
            matrix[state, state] += degrade
        if state < num_states - 2:
            matrix[state, state + 2] += skip
        else:
            matrix[state, state] += skip
    return matrix / matrix.sum(axis=1, keepdims=True)


def _distance_observation_tensor(num_states: int, accuracy: np.ndarray) -> np.ndarray:
    tensor = np.zeros((accuracy.size, num_states, num_states), dtype=float)
    for arm_index, diagonal in enumerate(accuracy):
        for state in range(num_states):
            weights = np.array(
                [0.0 if observation == state else 1.0 / abs(observation - state) for observation in range(num_states)],
                dtype=float,
            )
            weights /= weights.sum()
            tensor[arm_index, state] = (1.0 - diagonal) * weights
            tensor[arm_index, state, state] = diagonal
    return tensor


def _identity_observation_tensor(num_arms: int, num_states: int) -> np.ndarray:
    identity = np.eye(num_states, dtype=float)
    return np.repeat(identity[None, :, :], num_arms, axis=0)


def _regularized_probability_matrix(
    base: np.ndarray,
    rng: np.random.Generator,
    *,
    mix: float,
    jitter: float,
) -> np.ndarray:
    matrix = np.asarray(base, dtype=float).copy()
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    noise = rng.dirichlet(np.full(matrix.shape[1], 1.5), size=matrix.shape[0])
    matrix = (1.0 - mix) * matrix + mix * noise
    row_scale = rng.uniform(1.0 - jitter, 1.0 + jitter, size=matrix.shape)
    matrix = np.maximum(matrix * row_scale, 1e-5)
    return matrix / matrix.sum(axis=1, keepdims=True)


def _banded_shift_matrix(
    num_states: int,
    *,
    up1: float,
    up2: float,
    down1: float,
    down2: float,
) -> np.ndarray:
    matrix = np.zeros((num_states, num_states), dtype=float)
    for state in range(num_states):
        mass = 0.0
        if state < num_states - 1:
            matrix[state, state + 1] += up1
            mass += up1
        if state < num_states - 2:
            matrix[state, state + 2] += up2
            mass += up2
        if state > 0:
            matrix[state, state - 1] += down1
            mass += down1
        if state > 1:
            matrix[state, state - 2] += down2
            mass += down2
        matrix[state, state] += max(0.0, 1.0 - mass)
    return matrix / matrix.sum(axis=1, keepdims=True)


def _centered_belief(num_states: int, *, center: int, spread: float) -> np.ndarray:
    states = np.arange(num_states, dtype=float)
    weights = np.exp(-0.5 * ((states - float(center)) / float(spread)) ** 2)
    return weights / weights.sum()


def _reward_matrix_from_row(
    reward_row: np.ndarray,
    rng: np.random.Generator,
    *,
    scale: float,
    obs_noise: float,
) -> np.ndarray:
    row = np.asarray(reward_row, dtype=float)
    state_scale = rng.uniform(0.992, 1.008, size=row.shape[0])
    observation_shift = rng.normal(0.0, obs_noise, size=row.shape[0])
    interaction = rng.normal(0.0, 0.35 * obs_noise, size=(row.shape[0], row.shape[0]))
    reward = scale * row[:, None] * state_scale[:, None]
    reward = reward + observation_shift[None, :] + interaction
    return np.maximum(reward, 0.0)


def _noisy_belief(
    center: Sequence[float],
    rng: np.random.Generator,
    *,
    strength: float,
) -> np.ndarray:
    concentration = np.maximum(np.asarray(center, dtype=float), 1e-4) * strength
    return rng.dirichlet(concentration)


def _matmul_rows(beliefs: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    return np.einsum("nd,nde->ne", beliefs, matrices)


def _sample_states_from_beliefs(beliefs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _sample_categorical_rows(beliefs, rng)


def _sample_categorical_rows(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    draws = rng.random(probabilities.shape[0])[:, None]
    return np.sum(draws > cumulative, axis=1).astype(int)


def _as_probability_tensor(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional tensor.")
    if np.any(array < -1e-12):
        raise ValueError(f"{name} contains negative probabilities.")
    array = np.maximum(array, 0.0)
    row_sums = array.sum(axis=2)
    if not np.allclose(row_sums, 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} rows must sum to one.")
    return array


def _state_quality(num_states: int) -> np.ndarray:
    cached = _STATE_QUALITY_CACHE.get(int(num_states))
    if cached is not None:
        return cached
    if num_states == 1:
        cached = np.ones(1, dtype=float)
    else:
        cached = np.linspace(0.0, 1.0, num_states, dtype=float)
    _STATE_QUALITY_CACHE[int(num_states)] = cached
    return cached


def _cached_state_rewards(instance: MultiStateBatchInstance) -> np.ndarray:
    cache_key = _instance_cache_key(instance)
    cached = _STATE_REWARD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cached = np.sum(instance.E1 * instance.R, axis=2)
    _STATE_REWARD_CACHE[cache_key] = cached
    return cached


def _instance_cache_key(instance: MultiStateBatchInstance) -> _INSTANCE_CACHE_KEY:
    return (
        id(instance),
        int(instance.num_arms),
        int(instance.num_states),
        int(instance.num_observations),
        str(instance.name),
    )


def _cached_feedback_likelihoods(
    instance: MultiStateBatchInstance,
    arm_index: int,
    *,
    active: bool,
) -> tuple[np.ndarray, ...]:
    cache_key = (_instance_cache_key(instance), int(arm_index))
    if active:
        cached = _ACTIVE_FEEDBACK_LIKELIHOOD_CACHE.get(cache_key)
        if cached is not None:
            return cached
        likelihoods: list[np.ndarray] = []
        for observation in range(instance.num_observations):
            rounded_rewards = np.round(instance.R[arm_index, :, observation], 12)
            for reward in np.unique(rounded_rewards):
                mask = rounded_rewards == reward
                likelihood = instance.E1[arm_index, :, observation] * mask
                if np.any(likelihood > 1e-14):
                    likelihoods.append(np.asarray(likelihood, dtype=float))
        cached = tuple(likelihoods)
        _ACTIVE_FEEDBACK_LIKELIHOOD_CACHE[cache_key] = cached
        return cached
    cached = _PASSIVE_FEEDBACK_LIKELIHOOD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if instance.E0 is None:
        cached = tuple()
    else:
        cached = tuple(
            np.asarray(instance.E0[arm_index, :, observation], dtype=float)
            for observation in range(instance.num_observations)
        )
    _PASSIVE_FEEDBACK_LIKELIHOOD_CACHE[cache_key] = cached
    return cached


def _regular_block_reward_row(num_states: int) -> np.ndarray:
    top_count = max(1, int(round(0.2 * num_states)))
    bottom_count = max(1, int(round(0.2 * num_states)))
    if top_count + bottom_count > num_states:
        bottom_count = max(1, num_states - top_count)
    reward = np.full(num_states, 3.2, dtype=float)
    reward[:top_count] = 5.5
    reward[num_states - bottom_count :] = 0.1
    return reward


def _normalized_entropy(belief: np.ndarray) -> float:
    clipped = np.clip(np.asarray(belief, dtype=float), 1e-12, 1.0)
    entropy = float(-np.sum(clipped * np.log(clipped)))
    return entropy / np.log(clipped.size)


def _normalize(belief: np.ndarray) -> np.ndarray:
    b = np.maximum(np.asarray(belief, dtype=float), 0.0)
    return b / b.sum()


def _aggregate_outcomes(
    outcomes: Sequence[tuple[float, np.ndarray]],
) -> tuple[tuple[float, np.ndarray], ...]:
    aggregated: dict[tuple[float, ...], tuple[float, np.ndarray]] = {}
    for probability, belief in outcomes:
        key = tuple(float(x) for x in np.round(belief, 12))
        if key in aggregated:
            old_probability, old_belief = aggregated[key]
            total_probability = old_probability + probability
            merged = (old_probability * old_belief + probability * belief) / total_probability
            aggregated[key] = (total_probability, _normalize(merged))
        else:
            aggregated[key] = (float(probability), belief.copy())
    return tuple(aggregated.values())
