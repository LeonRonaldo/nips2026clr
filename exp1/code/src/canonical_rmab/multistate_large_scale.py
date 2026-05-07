from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MultiStateBatchInstance:
    """Vectorized canonical RMAB instance loaded from anonymous matrix files."""

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
            if e0.shape != e1.shape:
                raise ValueError(f"E0 must have shape {e1.shape}, got {e0.shape}.")

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


class OurLookaheadIndexPolicy:
    """Paper-level pseudocode for the proposed local-score index.

    This class documents the implementation recipe from the paper without
    exposing the optimized recurrence.

    Pseudocode:
        Input:
            current beliefs b_1(t), ..., b_N(t)
            budget K
            selected multiplier sequence lambda*
            tail values U_i,t+1^{lambda*}

        Offline tail construction:
            choose lambda* in the admissible price family Lambda by minimizing

                sum_i U_i,0^lambda(b_i(0))
                + K * sum_{tau=0}^{T-1} beta^tau * lambda_tau

            for each arm i and time t = T-1, ..., 0:
                U_i,T^lambda(b) = 0
                U_i,t^lambda(b) =
                    max over a in {0,1} of
                        r_i(b,a)
                        - lambda_t * a
                        + beta * E[U_i,t+1^lambda(B_i^+) | b,a]

        Online score evaluation:
            for each arm i:
                q_i(1) =
                    r_i(b_i(t),1)
                    + beta * E[U_i,t+1^{lambda*}(B_i^+) | b_i(t),1]

                q_i(0) =
                    r_i(b_i(t),0)
                    + beta * E[U_i,t+1^{lambda*}(B_i^+) | b_i(t),0]

                I_i = q_i(1) - q_i(0)

            activate any K arms with largest I_i values
            after feedback, update each belief by Bayes rule and transition
            prediction.

    The anonymous source gives this algorithmic specification only.  It omits
    the private code that constructs, caches, and interpolates the tail-value
    objects used in the reported implementation.
    """

    def __init__(self, lookahead: int = 2, cache_precision: int = 10) -> None:
        self.lookahead = int(lookahead)
        self.cache_precision = int(cache_precision)

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: MultiStateBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del beliefs, instance, time, rng
        raise NotImplementedError("Proposed index is specified as pseudocode in this anonymous source.")


class OurHybridTailIndexPolicy:
    """Paper-level pseudocode for the full-tail H=1 priority policy.

    This is the policy described by the paper's H=1 top-K equivalence.

    Pseudocode:
        Input:
            component models {P_i^a, L_i^a, r_i}
            current joint belief b(t) = (b_1(t), ..., b_N(t))
            horizon T, current time t, budget K
            selected multiplier sequence lambda*

        Step 1: relaxed single-component tail.
            for each arm i:
                set U_i,T^{lambda*}(b) = 0
                for s = T-1, ..., t+1:
                    U_i,s^{lambda*}(b) =
                        max over a in {0,1} of
                            r_i(b,a)
                            - lambda*_s * a
                            + beta * E[
                                U_i,s+1^{lambda*}(B_i^+)
                                | b,a
                              ]

        Step 2: separable Lagrangian tail bound.
            C_s^{lambda*} =
                sum_{tau=s}^{T-1} beta^{tau-s} lambda*_tau K

            Z_s^{0,lambda*}(b_1,...,b_N) =
                sum_i U_i,s^{lambda*}(b_i) + C_s^{lambda*}

        Step 3: one exact budget-constrained backup.
            Because Z_{t+1}^{0,lambda*} is separable, the H=1 action-value
            decomposes as

                Q_t^{1,lambda*}(b,a)
                = beta C_{t+1}^{lambda*}
                  + sum_i q_i,t^{lambda*}(b_i,a_i)

            where

                q_i,t^{lambda*}(b_i,a_i)
                = r_i(b_i,a_i)
                  + beta * E[
                      U_i,t+1^{lambda*}(B_i^+)
                      | b_i,a_i
                    ].

        Step 4: exact top-K optimizer.
            for each arm i:
                I_i,t^{lambda*}(b_i) =
                    q_i,t^{lambda*}(b_i,1)
                    - q_i,t^{lambda*}(b_i,0)

            return any feasible action selecting K largest I_i,t values.

        Step 5: filtering.
            after active or passive feedback, update each b_i by the paper's
            Bayes posterior over the pre-transition state, then multiply by
            P_i^a to obtain B_i^+.

    The source intentionally contains the pseudocode and public interface, not
    the optimized recurrence/caching implementation used to produce the paper's
    reported numbers.
    """

    def __init__(
        self,
        prefix_depth: int = 1,
        tail_horizon: int = 0,
        tail_mode: str = "commitment_envelope",
        resource_price: float | None = None,
        candidate_count: int | None = None,
        cache_precision: int = 8,
        max_consecutive: int | None = None,
        cooldown_steps: int = 0,
    ) -> None:
        self.prefix_depth = int(prefix_depth)
        self.tail_horizon = int(tail_horizon)
        self.tail_mode = str(tail_mode)
        self.resource_price = None if resource_price is None else float(resource_price)
        self.candidate_count = None if candidate_count is None else int(candidate_count)
        self.cache_precision = int(cache_precision)
        self.max_consecutive = None if max_consecutive is None else int(max_consecutive)
        self.cooldown_steps = int(cooldown_steps)

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: MultiStateBatchInstance,
        time: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del beliefs, instance, time, rng
        raise NotImplementedError("Full-tail index is specified as pseudocode in this anonymous source.")


class OurMeanTrajectoryIndexPolicy(OurHybridTailIndexPolicy):
    """Paper-level pseudocode shell for an auxiliary trajectory-index variant."""

    def __init__(
        self,
        planning_horizon: int = 8,
        max_consecutive: int | None = None,
        cooldown_steps: int = 0,
    ) -> None:
        super().__init__(
            prefix_depth=1,
            tail_horizon=planning_horizon,
            max_consecutive=max_consecutive,
            cooldown_steps=cooldown_steps,
        )
        self.planning_horizon = int(planning_horizon)


class RolloutMyopicMultiStatePolicy:
    """Monte Carlo rollout around myopic candidate actions."""

    def __init__(
        self,
        rollout_horizon: int = 4,
        rollout_samples: int = 1000,
        candidate_actions: int | None = None,
        seed: int = 0,
    ) -> None:
        if rollout_horizon <= 0:
            raise ValueError("rollout_horizon must be positive.")
        if rollout_samples <= 0:
            raise ValueError("rollout_samples must be positive.")
        if candidate_actions is not None and candidate_actions <= 0:
            raise ValueError("candidate_actions must be positive.")
        self.rollout_horizon = int(rollout_horizon)
        self.rollout_samples = int(rollout_samples)
        self.candidate_actions = None if candidate_actions is None else int(candidate_actions)
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        beliefs: np.ndarray,
        instance: MultiStateBatchInstance,
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
        instance: MultiStateBatchInstance,
        beliefs: np.ndarray,
        scores: np.ndarray,
    ) -> list[np.ndarray]:
        base = top_k(scores, instance.budget)
        candidates = [base]
        candidate_limit = instance.num_arms if self.candidate_actions is None else self.candidate_actions
        if candidate_limit == 1:
            return candidates

        base_set = set(int(index) for index in base)
        entropy = np.array([_normalized_entropy(belief) for belief in beliefs], dtype=float)
        active_next_beliefs = _matmul_rows(beliefs, instance.P1)
        two_step_scores = scores + instance.discount * expected_active_rewards(instance, active_next_beliefs)
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
                if len(add_pool) >= candidate_limit * 2:
                    break
            if len(add_pool) >= candidate_limit * 2:
                break

        for add_arm in add_pool:
            for drop_arm in drop_order:
                trial = np.array(sorted((base_set - {drop_arm}) | {add_arm}), dtype=int)
                if not any(np.array_equal(trial, candidate) for candidate in candidates):
                    candidates.append(trial)
                    if len(candidates) >= candidate_limit:
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
            action = first_action if step == 0 else top_k(expected_active_rewards(instance, sim_beliefs), instance.budget)
            reward, states, sim_beliefs = sample_public_step(instance, sim_beliefs, states, action, rng)
            total += discount * reward
            discount *= instance.discount
        return float(total)


def sample_public_step(
    instance: MultiStateBatchInstance,
    beliefs: np.ndarray,
    states: np.ndarray,
    active_arms: Sequence[int],
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    active = validate_action(active_arms, instance)
    active_mask = np.zeros(instance.num_arms, dtype=bool)
    active_mask[active] = True
    passive_idx = np.flatnonzero(~active_mask)

    next_beliefs = np.empty_like(beliefs)
    next_states = np.empty_like(states)
    rewards = np.zeros(instance.num_arms, dtype=float)

    if active.size:
        x = states[active]
        observations = _sample_categorical_rows(instance.E1[active, x, :], rng)
        rewards[active] = instance.R[active, x, observations]
        next_beliefs[active] = _active_next_beliefs(
            instance,
            beliefs[active],
            active,
            observations,
            rewards[active],
        )
        next_states[active] = _sample_categorical_rows(instance.P1[active, x, :], rng)

    if passive_idx.size:
        x = states[passive_idx]
        if instance.E0 is None:
            next_beliefs[passive_idx] = _matmul_rows(beliefs[passive_idx], instance.P0[passive_idx])
        else:
            observations = _sample_categorical_rows(instance.E0[passive_idx, x, :], rng)
            next_beliefs[passive_idx] = _passive_next_beliefs(
                instance,
                beliefs[passive_idx],
                passive_idx,
                observations,
            )
        next_states[passive_idx] = _sample_categorical_rows(instance.P0[passive_idx, x, :], rng)

    return float(rewards.sum()), next_states, next_beliefs


def expected_active_rewards(
    instance: MultiStateBatchInstance,
    beliefs: np.ndarray,
    arm_index: int | None = None,
    *,
    source_arm: int | None = None,
) -> np.ndarray | float:
    state_rewards = np.sum(instance.E1 * instance.R, axis=2)
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
    """Pseudocode for the public feedback expansion used inside expectations.

    For each feedback event (o,y) with positive probability:
        likelihood[x] = L_i^a(o,y | x)
        probability = sum_x b[x] * likelihood[x]
        posterior[x] = b[x] * likelihood[x] / probability
        next_belief[x'] = sum_x posterior[x] * P_i^a[x,x']
        emit (probability, next_belief)

    The concrete enumeration is omitted from the anonymous code because it is
    part of the private recurrence implementation.
    """
    del instance, belief, arm_index, active
    raise NotImplementedError("Feedback outcome expansion is specified as pseudocode in this anonymous source.")


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
    likelihood *= np.isclose(instance.R[idx, :, observations], rewards[:, None], atol=1e-12, rtol=0.0)
    denom = np.sum(beliefs * likelihood, axis=1, keepdims=True)
    posterior = beliefs * likelihood / denom
    return _matmul_rows(posterior, instance.P1[idx])


def _passive_next_beliefs(
    instance: MultiStateBatchInstance,
    beliefs: np.ndarray,
    idx: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    if instance.E0 is None:
        raise ValueError("passive observations require E0.")
    likelihood = instance.E0[idx, :, observations]
    denom = np.sum(beliefs * likelihood, axis=1, keepdims=True)
    posterior = beliefs * likelihood / denom
    return _matmul_rows(posterior, instance.P0[idx])


def _matmul_rows(beliefs: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    return np.einsum("nd,nde->ne", beliefs, matrices)


def _sample_states_from_beliefs(beliefs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _sample_categorical_rows(beliefs, rng)


def _sample_categorical_rows(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    draws = rng.random(probabilities.shape[0])[:, None]
    return np.sum(draws > cumulative, axis=1).astype(int)


def _normalized_entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0.0]
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(probabilities.size))


def _as_probability_tensor(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional tensor.")
    if np.any(array < -1e-12):
        raise ValueError(f"{name} contains negative probabilities.")
    array = np.maximum(array, 0.0)
    row_sums = array.sum(axis=2, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"{name} contains rows with zero probability mass.")
    return array / row_sums
