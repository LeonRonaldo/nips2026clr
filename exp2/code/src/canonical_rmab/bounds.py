from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product

import numpy as np

from .belief import (
    active_belief_update,
    expected_active_reward,
    passive_belief_update,
)
from .model import ArmModel, RMABInstance


@dataclass(frozen=True)
class FeedbackOutcome:
    probability: float
    next_belief: np.ndarray
    observation: int | None
    reward: float


@dataclass(frozen=True)
class ExactJointResult:
    value: float
    root_action: tuple[int, ...]
    states_evaluated: int
    action_count: int


@dataclass(frozen=True)
class SingleArmLagrangianResult:
    value: float
    states_evaluated: int


@dataclass(frozen=True)
class LagrangianBoundResult:
    upper_bound: float
    lambdas: np.ndarray
    arm_values: tuple[float, ...]
    constant: float
    states_evaluated: int


@dataclass(frozen=True)
class ScalarLagrangianSearchResult:
    best: LagrangianBoundResult
    candidates: tuple[LagrangianBoundResult, ...]


@dataclass(frozen=True)
class HybridLagrangianTailBoundResult:
    upper_bound: float
    expansion_depth: int
    lambdas: np.ndarray
    prefix_states_evaluated: int
    tail_states_evaluated: int
    action_count: int


@dataclass(frozen=True)
class FullyObservedBoundResult:
    upper_bound: float
    states_evaluated: int
    action_count: int


@dataclass(frozen=True)
class QMDPBoundResult:
    upper_bound: float
    states_evaluated: int
    action_count: int


def exact_joint_value(
    instance: RMABInstance,
    horizon: int | None = None,
    round_decimals: int = 12,
) -> ExactJointResult:
    """Exact finite-horizon value on the reachable joint belief tree.

    This is intended for tiny reference cases only. It enumerates feasible
    global actions and all public feedback outcomes.
    """
    total_horizon = instance.horizon if horizon is None else int(horizon)
    if total_horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    actions = enumerate_budget_actions(instance.num_arms, instance.budget, instance.exactly_k)
    state_counter = 0

    def beliefs_from_key(key: tuple[tuple[float, ...], ...]) -> tuple[np.ndarray, ...]:
        return tuple(np.array(local, dtype=float) for local in key)

    @lru_cache(maxsize=None)
    def value(time: int, key: tuple[tuple[float, ...], ...]) -> tuple[float, tuple[int, ...]]:
        nonlocal state_counter
        state_counter += 1
        if time >= total_horizon:
            return 0.0, ()

        beliefs = beliefs_from_key(key)
        best_value = -np.inf
        best_action: tuple[int, ...] = actions[0]
        for action in actions:
            active = set(action)
            immediate = sum(
                expected_active_reward(beliefs[index], arm)
                for index, arm in enumerate(instance.arms)
                if index in active
            )
            local_outcomes = [
                active_feedback_outcomes(beliefs[index], arm)
                if index in active
                else passive_feedback_outcomes(beliefs[index], arm)
                for index, arm in enumerate(instance.arms)
            ]
            continuation = 0.0
            for outcome_tuple in product(*local_outcomes):
                probability = float(np.prod([outcome.probability for outcome in outcome_tuple]))
                if probability <= 0.0:
                    continue
                next_key = _beliefs_key(
                    tuple(outcome.next_belief for outcome in outcome_tuple),
                    round_decimals,
                )
                next_value, _ = value(time + 1, next_key)
                continuation += probability * next_value
            candidate = float(immediate + instance.discount * continuation)
            if candidate > best_value:
                best_value = candidate
                best_action = tuple(action)
        return best_value, best_action

    initial_key = _beliefs_key(instance.initial_beliefs, round_decimals)
    root_value, root_action = value(0, initial_key)
    return ExactJointResult(
        value=float(root_value),
        root_action=tuple(root_action),
        states_evaluated=state_counter,
        action_count=len(actions),
    )


def lagrangian_upper_bound(
    instance: RMABInstance,
    lambdas: Sequence[float],
    horizon: int | None = None,
    round_decimals: int = 12,
) -> LagrangianBoundResult:
    """Finite-horizon Lagrangian relaxation upper bound.

    The single-arm subproblem maximizes

        r(b,a) - lambda_t a + discount * E[V_{t+1}(b+)].

    For exactly-K constraints the bound is

        sum_i U_i^lambda(b_i,0) + sum_t discount^t lambda_t K.
    """
    total_horizon = instance.horizon if horizon is None else int(horizon)
    lambdas_array = np.asarray(lambdas, dtype=float)
    if lambdas_array.shape != (total_horizon,):
        raise ValueError(f"lambdas must have shape ({total_horizon},).")
    if not instance.exactly_k and np.any(lambdas_array < -1e-12):
        raise ValueError("at-most-K Lagrangian multipliers must be nonnegative.")

    arm_results = tuple(
        single_arm_lagrangian_value(
            arm,
            belief,
            horizon=total_horizon,
            discount=instance.discount,
            lambdas=lambdas_array,
            round_decimals=round_decimals,
        )
        for arm, belief in zip(instance.arms, instance.initial_beliefs)
    )
    discount_weights = instance.discount ** np.arange(total_horizon, dtype=float)
    constant = float(instance.budget * np.dot(discount_weights, lambdas_array))
    upper_bound = float(sum(result.value for result in arm_results) + constant)
    return LagrangianBoundResult(
        upper_bound=upper_bound,
        lambdas=lambdas_array.copy(),
        arm_values=tuple(float(result.value) for result in arm_results),
        constant=constant,
        states_evaluated=sum(result.states_evaluated for result in arm_results),
    )


def scalar_lagrangian_grid_search(
    instance: RMABInstance,
    theta_grid: Iterable[float],
    horizon: int | None = None,
    round_decimals: int = 12,
) -> ScalarLagrangianSearchResult:
    """Search constant-in-time multipliers `lambda_t = theta`."""
    total_horizon = instance.horizon if horizon is None else int(horizon)
    candidates = tuple(
        lagrangian_upper_bound(
            instance,
            lambdas=np.full(total_horizon, float(theta)),
            horizon=total_horizon,
            round_decimals=round_decimals,
        )
        for theta in theta_grid
    )
    if not candidates:
        raise ValueError("theta_grid must be nonempty.")
    best = min(candidates, key=lambda result: result.upper_bound)
    return ScalarLagrangianSearchResult(best=best, candidates=candidates)


def hybrid_lagrangian_tail_bound(
    instance: RMABInstance,
    lambdas: Sequence[float],
    expansion_depth: int,
    horizon: int | None = None,
    round_decimals: int = 12,
) -> HybridLagrangianTailBoundResult:
    """Upper bound from exact joint expansion with a Lagrangian tail.

    The first ``expansion_depth`` periods use the true joint action constraint
    and public feedback tree. The remaining finite-horizon tail is bounded by
    the single-arm Lagrangian relaxation from the reached belief state.
    """
    total_horizon = instance.horizon if horizon is None else int(horizon)
    if total_horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    depth = int(expansion_depth)
    if depth < 0:
        raise ValueError("expansion_depth must be nonnegative.")
    depth = min(depth, total_horizon)
    lambdas_array = np.asarray(lambdas, dtype=float)
    if lambdas_array.shape != (total_horizon,):
        raise ValueError(f"lambdas must have shape ({total_horizon},).")
    if not instance.exactly_k and np.any(lambdas_array < -1e-12):
        raise ValueError("at-most-K Lagrangian multipliers must be nonnegative.")

    actions = enumerate_budget_actions(instance.num_arms, instance.budget, instance.exactly_k)
    prefix_counter = 0
    tail_counter = 0

    def beliefs_from_key(key: tuple[tuple[float, ...], ...]) -> tuple[np.ndarray, ...]:
        return tuple(np.array(local, dtype=float) for local in key)

    @lru_cache(maxsize=None)
    def tail_value(time: int, key: tuple[tuple[float, ...], ...]) -> float:
        nonlocal tail_counter
        remaining = total_horizon - time
        if remaining <= 0:
            return 0.0
        beliefs = beliefs_from_key(key)
        arm_results = tuple(
            single_arm_lagrangian_value(
                arm,
                belief,
                horizon=remaining,
                discount=instance.discount,
                lambdas=lambdas_array[time:total_horizon],
                round_decimals=round_decimals,
            )
            for arm, belief in zip(instance.arms, beliefs)
        )
        tail_counter += sum(result.states_evaluated for result in arm_results)
        discount_weights = instance.discount ** np.arange(remaining, dtype=float)
        constant = float(instance.budget * np.dot(discount_weights, lambdas_array[time:total_horizon]))
        return float(sum(result.value for result in arm_results) + constant)

    @lru_cache(maxsize=None)
    def value(time: int, depth_left: int, key: tuple[tuple[float, ...], ...]) -> float:
        nonlocal prefix_counter
        prefix_counter += 1
        if time >= total_horizon:
            return 0.0
        if depth_left <= 0:
            return tail_value(time, key)

        beliefs = beliefs_from_key(key)
        best = -np.inf
        for action in actions:
            active = set(action)
            immediate = sum(
                expected_active_reward(beliefs[index], arm)
                for index, arm in enumerate(instance.arms)
                if index in active
            )
            local_outcomes = [
                active_feedback_outcomes(beliefs[index], arm)
                if index in active
                else passive_feedback_outcomes(beliefs[index], arm)
                for index, arm in enumerate(instance.arms)
            ]
            continuation = 0.0
            for outcome_tuple in product(*local_outcomes):
                probability = float(np.prod([outcome.probability for outcome in outcome_tuple]))
                if probability <= 0.0:
                    continue
                next_key = _beliefs_key(
                    tuple(outcome.next_belief for outcome in outcome_tuple),
                    round_decimals,
                )
                continuation += probability * value(time + 1, depth_left - 1, next_key)
            best = max(best, float(immediate + instance.discount * continuation))
        return best

    root = _beliefs_key(instance.initial_beliefs, round_decimals)
    upper_bound = value(0, depth, root)
    return HybridLagrangianTailBoundResult(
        upper_bound=float(upper_bound),
        expansion_depth=depth,
        lambdas=lambdas_array.copy(),
        prefix_states_evaluated=prefix_counter,
        tail_states_evaluated=tail_counter,
        action_count=len(actions),
    )


def fully_observed_mdp_bound(
    instance: RMABInstance,
    horizon: int | None = None,
) -> FullyObservedBoundResult:
    """Finite-horizon upper bound where the controller observes hidden states.

    This keeps the same hard budget and transition/reward model, but gives the
    policy the hidden state vector before action selection. It is therefore an
    information-relaxation upper bound on the partially observed problem.
    """
    total_horizon = instance.horizon if horizon is None else int(horizon)
    if total_horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    actions = enumerate_budget_actions(instance.num_arms, instance.budget, instance.exactly_k)
    state_ranges = [range(arm.num_states) for arm in instance.arms]
    state_counter = 0

    @lru_cache(maxsize=None)
    def value(time: int, state: tuple[int, ...]) -> float:
        nonlocal state_counter
        state_counter += 1
        if time >= total_horizon:
            return 0.0
        best = -np.inf
        for action in actions:
            active = set(action)
            immediate = 0.0
            local_transition_rows = []
            for arm_index, arm in enumerate(instance.arms):
                x = int(state[arm_index])
                if arm_index in active:
                    immediate += float(np.dot(arm.E1[x], arm.R[x]))
                    local_transition_rows.append(arm.P1[x])
                else:
                    local_transition_rows.append(arm.P0[x])
            continuation = 0.0
            for next_state in product(*state_ranges):
                probability = 1.0
                for arm_index, y in enumerate(next_state):
                    probability *= float(local_transition_rows[arm_index][y])
                if probability <= 0.0:
                    continue
                continuation += probability * value(time + 1, tuple(int(y) for y in next_state))
            best = max(best, float(immediate + instance.discount * continuation))
        return best

    upper_bound = 0.0
    for initial_state in product(*state_ranges):
        probability = 1.0
        for arm_index, x in enumerate(initial_state):
            probability *= float(instance.initial_beliefs[arm_index][x])
        if probability <= 0.0:
            continue
        upper_bound += probability * value(0, tuple(int(x) for x in initial_state))

    return FullyObservedBoundResult(
        upper_bound=float(upper_bound),
        states_evaluated=state_counter,
        action_count=len(actions),
    )


def qmdp_bound(
    instance: RMABInstance,
    horizon: int | None = None,
) -> QMDPBoundResult:
    """Finite-horizon QMDP upper bound.

    The root action is chosen from the public belief. After that action, the
    controller is granted full hidden-state observability for the remaining
    horizon. This is tighter than the root fully-observed MDP bound but remains
    an information-relaxation upper bound for the POMDP.
    """
    total_horizon = instance.horizon if horizon is None else int(horizon)
    if total_horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    actions = enumerate_budget_actions(instance.num_arms, instance.budget, instance.exactly_k)
    state_ranges = [range(arm.num_states) for arm in instance.arms]
    state_counter = 0

    @lru_cache(maxsize=None)
    def mdp_value(time: int, state: tuple[int, ...]) -> float:
        nonlocal state_counter
        state_counter += 1
        if time >= total_horizon:
            return 0.0
        best = -np.inf
        for action in actions:
            active = set(action)
            immediate = 0.0
            local_transition_rows = []
            for arm_index, arm in enumerate(instance.arms):
                x = int(state[arm_index])
                if arm_index in active:
                    immediate += float(np.dot(arm.E1[x], arm.R[x]))
                    local_transition_rows.append(arm.P1[x])
                else:
                    local_transition_rows.append(arm.P0[x])
            continuation = 0.0
            for next_state in product(*state_ranges):
                probability = 1.0
                for arm_index, y in enumerate(next_state):
                    probability *= float(local_transition_rows[arm_index][y])
                if probability <= 0.0:
                    continue
                continuation += probability * mdp_value(time + 1, tuple(int(y) for y in next_state))
            best = max(best, float(immediate + instance.discount * continuation))
        return best

    if total_horizon == 0:
        return QMDPBoundResult(upper_bound=0.0, states_evaluated=0, action_count=len(actions))

    best_root = -np.inf
    for action in actions:
        active = set(action)
        immediate = sum(
            expected_active_reward(instance.initial_beliefs[index], arm)
            for index, arm in enumerate(instance.arms)
            if index in active
        )
        next_marginals = []
        for arm_index, (belief, arm) in enumerate(zip(instance.initial_beliefs, instance.arms)):
            transition = arm.P1 if arm_index in active else arm.P0
            next_marginals.append(np.asarray(belief, dtype=float) @ transition)
        continuation = 0.0
        for next_state in product(*state_ranges):
            probability = 1.0
            for arm_index, y in enumerate(next_state):
                probability *= float(next_marginals[arm_index][y])
            if probability <= 0.0:
                continue
            continuation += probability * mdp_value(1, tuple(int(y) for y in next_state))
        best_root = max(best_root, float(immediate + instance.discount * continuation))

    return QMDPBoundResult(
        upper_bound=float(best_root),
        states_evaluated=state_counter,
        action_count=len(actions),
    )


def single_arm_lagrangian_value(
    arm: ArmModel,
    initial_belief: np.ndarray,
    *,
    horizon: int,
    discount: float,
    lambdas: Sequence[float],
    round_decimals: int = 12,
) -> SingleArmLagrangianResult:
    if horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    lambdas_array = np.asarray(lambdas, dtype=float)
    if lambdas_array.shape != (horizon,):
        raise ValueError(f"lambdas must have shape ({horizon},).")
    state_counter = 0

    @lru_cache(maxsize=None)
    def value(time: int, key: tuple[float, ...]) -> float:
        nonlocal state_counter
        state_counter += 1
        if time >= horizon:
            return 0.0
        belief = np.array(key, dtype=float)

        passive = 0.0
        for outcome in passive_feedback_outcomes(belief, arm):
            passive += outcome.probability * value(
                time + 1,
                _belief_key(outcome.next_belief, round_decimals),
            )
        q_passive = float(discount * passive)

        active = 0.0
        for outcome in active_feedback_outcomes(belief, arm):
            active += outcome.probability * value(
                time + 1,
                _belief_key(outcome.next_belief, round_decimals),
            )
        q_active = float(expected_active_reward(belief, arm) - lambdas_array[time] + discount * active)
        return max(q_passive, q_active)

    root = _belief_key(initial_belief, round_decimals)
    return SingleArmLagrangianResult(value=float(value(0, root)), states_evaluated=state_counter)


def active_feedback_outcomes(
    belief: np.ndarray,
    arm: ArmModel,
    probability_tol: float = 1e-14,
) -> tuple[FeedbackOutcome, ...]:
    outcomes: list[FeedbackOutcome] = []
    for observation in range(arm.num_active_observations):
        possible_rewards = sorted({float(value) for value in arm.R[:, observation]})
        for reward in possible_rewards:
            likelihood = arm.E1[:, observation] * np.isclose(
                arm.R[:, observation],
                reward,
                atol=1e-12,
                rtol=0.0,
            )
            probability = float(np.asarray(belief, dtype=float) @ likelihood)
            if probability <= probability_tol:
                continue
            next_belief, _ = active_belief_update(belief, arm, observation, reward)
            outcomes.append(
                FeedbackOutcome(
                    probability=probability,
                    next_belief=next_belief,
                    observation=observation,
                    reward=reward,
                )
            )
    _validate_outcome_probabilities(outcomes, "active")
    return tuple(outcomes)


def passive_feedback_outcomes(
    belief: np.ndarray,
    arm: ArmModel,
    probability_tol: float = 1e-14,
) -> tuple[FeedbackOutcome, ...]:
    if arm.E0 is None:
        next_belief, probability = passive_belief_update(belief, arm)
        return (
            FeedbackOutcome(
                probability=probability,
                next_belief=next_belief,
                observation=None,
                reward=0.0,
            ),
        )

    outcomes: list[FeedbackOutcome] = []
    for observation in range(arm.num_passive_observations):
        probability = float(np.asarray(belief, dtype=float) @ arm.E0[:, observation])
        if probability <= probability_tol:
            continue
        next_belief, _ = passive_belief_update(belief, arm, observation)
        outcomes.append(
            FeedbackOutcome(
                probability=probability,
                next_belief=next_belief,
                observation=observation,
                reward=0.0,
            )
        )
    _validate_outcome_probabilities(outcomes, "passive")
    return tuple(outcomes)


def enumerate_budget_actions(
    num_arms: int,
    budget: int,
    exactly_k: bool = True,
) -> tuple[tuple[int, ...], ...]:
    if budget < 0 or budget > num_arms:
        raise ValueError("budget must be between 0 and num_arms.")
    sizes = (budget,) if exactly_k else tuple(range(budget + 1))
    actions: list[tuple[int, ...]] = []
    for size in sizes:
        actions.extend(tuple(int(index) for index in combo) for combo in combinations(range(num_arms), size))
    if not actions:
        actions.append(())
    return tuple(actions)


def _belief_key(belief: np.ndarray, decimals: int) -> tuple[float, ...]:
    rounded = np.round(np.asarray(belief, dtype=float), decimals=decimals)
    rounded = rounded / rounded.sum()
    return tuple(float(value) for value in rounded)


def _beliefs_key(beliefs: Sequence[np.ndarray], decimals: int) -> tuple[tuple[float, ...], ...]:
    return tuple(_belief_key(belief, decimals) for belief in beliefs)


def _validate_outcome_probabilities(outcomes: Sequence[FeedbackOutcome], name: str) -> None:
    total = float(sum(outcome.probability for outcome in outcomes))
    if not np.isclose(total, 1.0, atol=1e-10, rtol=1e-10):
        raise RuntimeError(f"{name} feedback probabilities sum to {total}, not one.")
