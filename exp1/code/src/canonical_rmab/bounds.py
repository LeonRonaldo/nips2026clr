from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

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
    """Paper-level pseudocode for exact joint finite-horizon belief DP.

    Pseudocode:
        V_T(b_1,...,b_N) = 0
        for t = T-1, ..., 0:
            for each reachable joint belief b:
                for each feasible activation vector a with sum_i a_i = K:
                    immediate = sum_i r_i(b_i, a_i)
                    continuation = 0
                    for each joint public feedback f with positive probability:
                        b_plus = tau(b, a, f)
                        continuation += Pr(f | b,a) * V_{t+1}(b_plus)
                    Q(a) = immediate + beta * continuation
                V_t(b) = max_a Q(a)
                root_action = argmax_a Q(a) at the initial belief

    The anonymous source records the recursion but omits the executable tree
    enumeration used in the private experiment code.
    """
    del instance, horizon, round_decimals
    raise NotImplementedError("Exact joint recursion is specified as pseudocode in this anonymous source.")


def lagrangian_upper_bound(
    instance: RMABInstance,
    lambdas: Sequence[float],
    horizon: int | None = None,
    round_decimals: int = 12,
) -> LagrangianBoundResult:
    """Paper-level pseudocode for the separable Lagrangian upper bound.

    Pseudocode:
        for each arm i:
            U_i,T^lambda(b) = 0
            for t = T-1, ..., 0:
                U_i,t^lambda(b) =
                    max over a in {0,1} of
                        r_i(b,a)
                        - lambda_t * a
                        + beta * E[
                            U_i,t+1^lambda(B_i^+) | b,a
                          ]

        C_t^lambda = sum_{tau=t}^{T-1} beta^{tau-t} lambda_tau K
        Z_t^{0,lambda}(b_1,...,b_N) =
            sum_i U_i,t^lambda(b_i) + C_t^lambda

        return Z_0^{0,lambda}(initial_belief)
    """
    del instance, lambdas, horizon, round_decimals
    raise NotImplementedError("Lagrangian bound is specified as pseudocode in this anonymous source.")


def scalar_lagrangian_grid_search(
    instance: RMABInstance,
    theta_grid: Iterable[float],
    horizon: int | None = None,
    round_decimals: int = 12,
) -> ScalarLagrangianSearchResult:
    """Paper-level pseudocode for scalar multiplier selection.

    Pseudocode:
        candidates = []
        for theta in theta_grid:
            lambda_t = theta for all t
            evaluate Z_0^{0,lambda}(b(0))
            append result
        choose the theta with smallest initial Lagrangian upper bound
    """
    del instance, theta_grid, horizon, round_decimals
    raise NotImplementedError("Multiplier search is specified as pseudocode in this anonymous source.")


def hybrid_lagrangian_tail_bound(
    instance: RMABInstance,
    lambdas: Sequence[float],
    expansion_depth: int,
    horizon: int | None = None,
    round_decimals: int = 12,
) -> HybridLagrangianTailBoundResult:
    """Paper-level pseudocode for exact-prefix plus Lagrangian-tail bounds.

    Pseudocode:
        build Z^{0,lambda} from the separable single-arm Lagrangian tail
        for h = 1, ..., H:
            for t = T-1, ..., 0:
                Z_t^{h,lambda}(b) =
                    max over feasible a of
                        r(b,a)
                        + beta * E[
                            Z_{t+1}^{h-1,lambda}(B^+) | b,a
                          ]

        The resulting value is an upper bound and is no larger than the
        previous-depth bound.
    """
    del instance, lambdas, expansion_depth, horizon, round_decimals
    raise NotImplementedError("Hybrid tail bound is specified as pseudocode in this anonymous source.")


def fully_observed_mdp_bound(
    instance: RMABInstance,
    horizon: int | None = None,
) -> FullyObservedBoundResult:
    """Paper-level pseudocode for the fully observed information bound.

    Pseudocode:
        grant the controller the hidden state vector before each action
        run the finite-horizon budget-constrained MDP recursion over states
        average the root value over the initial belief distribution
    """
    del instance, horizon
    raise NotImplementedError("Fully observed bound is specified as pseudocode in this anonymous source.")


def qmdp_bound(
    instance: RMABInstance,
    horizon: int | None = None,
) -> QMDPBoundResult:
    """Paper-level pseudocode for the QMDP information bound.

    Pseudocode:
        choose the root action from the public belief
        after that root action, relax the problem by revealing hidden states
        evaluate the remaining finite-horizon fully observed MDP tail
    """
    del instance, horizon
    raise NotImplementedError("QMDP bound is specified as pseudocode in this anonymous source.")


def single_arm_lagrangian_value(
    arm: ArmModel,
    initial_belief: np.ndarray,
    *,
    horizon: int,
    discount: float,
    lambdas: Sequence[float],
    round_decimals: int = 12,
) -> SingleArmLagrangianResult:
    """Paper-level pseudocode for the single-arm relaxed subproblem.

    Pseudocode:
        U_T(b) = 0
        for t = T-1, ..., 0:
            passive =
                r_i(b,0)
                + beta * E[U_{t+1}(B_i^+) | b,0]

            active =
                r_i(b,1)
                - lambda_t
                + beta * E[U_{t+1}(B_i^+) | b,1]

            U_t(b) = max(passive, active)
    """
    del arm, initial_belief, horizon, discount, lambdas, round_decimals
    raise NotImplementedError("Single-arm relaxed subproblem is specified as pseudocode in this anonymous source.")


def active_feedback_outcomes(
    belief: np.ndarray,
    arm: ArmModel,
    probability_tol: float = 1e-14,
) -> tuple[FeedbackOutcome, ...]:
    """Paper-level pseudocode for active public-feedback expansion.

    Pseudocode:
        for each active feedback pair (observation o, reward y):
            likelihood[x] = L_i^1(o,y | x)
            probability = sum_x b[x] likelihood[x]
            if probability > tolerance:
                eta[x] = b[x] likelihood[x] / probability
                next_belief[x'] = sum_x eta[x] P_i^1[x,x']
                emit probability, next_belief, o, y
    """
    del belief, arm, probability_tol
    raise NotImplementedError("Active-feedback expansion is specified as pseudocode in this anonymous source.")


def passive_feedback_outcomes(
    belief: np.ndarray,
    arm: ArmModel,
    probability_tol: float = 1e-14,
) -> tuple[FeedbackOutcome, ...]:
    """Paper-level pseudocode for passive public-feedback expansion.

    Pseudocode:
        if passive feedback is absent:
            emit probability 1 and next_belief = b P_i^0
        otherwise, for each passive observation o:
            likelihood[x] = L_i^0(o,0 | x)
            probability = sum_x b[x] likelihood[x]
            if probability > tolerance:
                eta[x] = b[x] likelihood[x] / probability
                next_belief[x'] = sum_x eta[x] P_i^0[x,x']
                emit probability, next_belief, o, reward 0
    """
    del belief, arm, probability_tol
    raise NotImplementedError("Passive-feedback expansion is specified as pseudocode in this anonymous source.")


def enumerate_budget_actions(
    num_arms: int,
    budget: int,
    exactly_k: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Return budget-feasible action tuples; kept because it is not paper-specific."""
    if budget < 0 or budget > num_arms:
        raise ValueError("budget must be between 0 and num_arms.")
    if exactly_k:
        sizes = (budget,)
    else:
        sizes = tuple(range(budget + 1))
    actions: list[tuple[int, ...]] = []
    for size in sizes:
        actions.extend(_combinations(range(num_arms), size))
    return tuple(actions or [()])


def _combinations(values: range, size: int) -> list[tuple[int, ...]]:
    if size == 0:
        return [()]
    if size > len(values):
        return []
    result: list[tuple[int, ...]] = []

    def visit(start: int, chosen: list[int]) -> None:
        if len(chosen) == size:
            result.append(tuple(chosen))
            return
        for index in range(start, len(values)):
            chosen.append(int(values[index]))
            visit(index + 1, chosen)
            chosen.pop()

    visit(0, [])
    return result
