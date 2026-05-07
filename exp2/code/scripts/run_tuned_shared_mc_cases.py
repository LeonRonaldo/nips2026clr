from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from canonical_rmab.multistate_large_scale import (
    ExpectedRolloutMyopicMultiStatePolicy,
    MultiStateBatchInstance,
    MultiStateBatchSimulator,
    MyopicMultiStatePolicy,
    OurHybridTailIndexPolicy,
    SimpleVFAMultiStatePolicy,
    _banded_shift_matrix,
    _centered_belief,
    _distance_observation_tensor,
    _identity_observation_tensor,
    _noisy_belief,
    _regularized_probability_matrix,
)


POLICY_ORDER = ["our_index", "rollout_myopic", "simple_vfa", "myopic"]
POLICY_LABELS = {
    "our_index": "Ours (tail index)",
    "rollout_myopic": "Rollout-myopic",
    "simple_vfa": "VFA",
    "myopic": "Myopic",
}
POLICY_COLORS = {
    "our_index": "#1f77b4",
    "rollout_myopic": "#d62728",
    "simple_vfa": "#2ca02c",
    "myopic": "#7f7f7f",
}
POLICY_STYLES = {
    "our_index": "-",
    "rollout_myopic": "--",
    "simple_vfa": "-.",
    "myopic": ":",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sampled-MC plots for tuned shared-R/shared-E three-case instances."
    )
    parser.add_argument("--num-arms", type=int, default=100)
    parser.add_argument("--num-states", type=int, default=8)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tuned_shared_mc_cases"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, object] = {}
    all_metrics: dict[str, object] = {}
    for case in ("case1", "case2", "case3"):
        instance = make_tuned_shared_instance(
            case=case,
            num_arms=args.num_arms,
            num_states=args.num_states,
            budget=args.budget,
            horizon=args.horizon,
            discount=args.discount,
            seed=args.seed,
        )
        case_results, case_metrics = run_case(instance, episodes=args.episodes, seed=args.seed)
        all_results[case] = case_results
        all_metrics[case] = case_metrics
        plot_case(
            case,
            case_results,
            case_metrics,
            args.output_dir / f"{case}_average_cumulative_per_step_reward_mc.png",
        )
        print(
            f"{case}: "
            f"ours={case_metrics['final']['our_index']:.4f}, "
            f"rollout={case_metrics['final']['rollout_myopic']:.4f}, "
            f"vfa={case_metrics['final']['simple_vfa']:.4f}, "
            f"myopic={case_metrics['final']['myopic']:.4f}, "
            f"gap={case_metrics['final_gap_pct_vs_best_baseline']:.2f}%"
        )

    (args.output_dir / "traces.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"OUTPUT_DIR: {args.output_dir}")


def make_tuned_shared_instance(
    *,
    case: str,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    discount: float,
    seed: int,
) -> MultiStateBatchInstance:
    if case == "case1":
        return _make_case1(
            num_arms=num_arms,
            num_states=num_states,
            budget=budget,
            horizon=horizon,
            discount=discount,
            seed=seed,
        )
    if case == "case2":
        return _make_case2(
            num_arms=num_arms,
            num_states=num_states,
            budget=budget,
            horizon=horizon,
            discount=discount,
            seed=seed,
        )
    if case == "case3":
        return _make_case3(
            num_arms=num_arms,
            num_states=num_states,
            budget=budget,
            horizon=horizon,
            discount=discount,
            seed=seed,
        )
    raise ValueError("case must be case1, case2, or case3.")


def _make_case1(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    discount: float,
    seed: int,
) -> MultiStateBatchInstance:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, num_states)
    reward_row = 5.30 + 1.85 * (1.0 - np.exp(-1.85 * grid)) + 6.50 * grid**5.50
    shared_R = _shared_reward_matrix(reward_row, rng, obs_amplitude=0.075, interaction_scale=0.018)
    return _assemble_instance(
        case="case1",
        num_arms=num_arms,
        num_states=num_states,
        budget=budget,
        horizon=horizon,
        discount=discount,
        seed=seed,
        shared_R=shared_R,
        E1=_identity_observation_tensor(num_arms, num_states),
        E0=None,
        dynamics={
            "project": ((0.006, 0.001, 0.130, 0.050), (0.340, 0.380, 0.003, 0.001), 0.27, 0.84),
            "rollout": ((0.024, 0.004, 0.060, 0.016), (0.320, 0.120, 0.035, 0.008), 0.35, 0.90),
            "vfa": ((0.024, 0.004, 0.070, 0.020), (0.160, 0.045, 0.052, 0.012), 0.43, 0.88),
            "cash": ((0.030, 0.006, 0.075, 0.022), (0.012, 0.002, 0.125, 0.038), 0.46, 0.88),
        },
    )


def _make_case2(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    discount: float,
    seed: int,
) -> MultiStateBatchInstance:
    rng = np.random.default_rng(seed + 17)
    grid = np.linspace(0.0, 1.0, num_states)
    reward_row = 10.30 + 1.70 * (1.0 - np.exp(-1.70 * grid)) + 6.80 * grid**5.70
    shared_R = _shared_reward_matrix(reward_row, rng, obs_amplitude=0.080, interaction_scale=0.020)
    E1 = _distance_observation_tensor(num_states, np.full(num_arms, 0.86, dtype=float))
    return _assemble_instance(
        case="case2",
        num_arms=num_arms,
        num_states=num_states,
        budget=budget,
        horizon=horizon,
        discount=discount,
        seed=seed + 17,
        shared_R=shared_R,
        E1=E1,
        E0=None,
        dynamics={
            "project": ((0.006, 0.001, 0.130, 0.050), (0.360, 0.400, 0.003, 0.001), 0.28, 0.84),
            "rollout": ((0.024, 0.004, 0.060, 0.016), (0.260, 0.070, 0.050, 0.010), 0.34, 0.90),
            "vfa": ((0.024, 0.004, 0.070, 0.020), (0.150, 0.040, 0.055, 0.012), 0.42, 0.88),
            "cash": ((0.028, 0.005, 0.080, 0.024), (0.012, 0.002, 0.130, 0.040), 0.45, 0.88),
        },
    )


def _make_case3(
    *,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    discount: float,
    seed: int,
) -> MultiStateBatchInstance:
    rng = np.random.default_rng(seed + 33)
    grid = np.linspace(0.0, 1.0, num_states)
    reward_row = 0.18 + 1.55 * (1.0 - np.exp(-1.45 * grid)) + 7.80 * grid**6.25
    shared_R = _shared_reward_matrix(reward_row, rng, obs_amplitude=0.090, interaction_scale=0.018)
    E1 = _distance_observation_tensor(num_states, np.full(num_arms, 0.89, dtype=float))
    E0 = _distance_observation_tensor(num_states, np.full(num_arms, 0.46, dtype=float))
    return _assemble_instance(
        case="case3",
        num_arms=num_arms,
        num_states=num_states,
        budget=budget,
        horizon=horizon,
        discount=discount,
        seed=seed + 33,
        shared_R=shared_R,
        E1=E1,
        E0=E0,
        dynamics={
            "project": ((0.006, 0.001, 0.140, 0.050), (0.380, 0.440, 0.003, 0.001), 0.26, 0.82),
            "rollout": ((0.020, 0.004, 0.070, 0.020), (0.240, 0.060, 0.055, 0.012), 0.32, 0.90),
            "vfa": ((0.020, 0.004, 0.075, 0.020), (0.140, 0.040, 0.055, 0.012), 0.40, 0.88),
            "cash": ((0.020, 0.004, 0.095, 0.030), (0.010, 0.002, 0.150, 0.050), 0.42, 0.88),
        },
    )


def _assemble_instance(
    *,
    case: str,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    discount: float,
    seed: int,
    shared_R: np.ndarray,
    E1: np.ndarray,
    E0: np.ndarray | None,
    dynamics: dict[str, tuple[tuple[float, float, float, float], tuple[float, float, float, float], float, float]],
) -> MultiStateBatchInstance:
    rng = np.random.default_rng(seed)
    project_arms = max(10, int(round(0.28 * num_arms)))
    rollout_arms = max(10, int(round(0.24 * num_arms)))
    vfa_arms = max(8, int(round(0.18 * num_arms)))
    if project_arms + rollout_arms + vfa_arms >= num_arms:
        project_arms = max(8, int(round(0.26 * num_arms)))
        rollout_arms = max(8, int(round(0.22 * num_arms)))
        vfa_arms = max(6, int(round(0.16 * num_arms)))
    cash_start = project_arms + rollout_arms + vfa_arms

    matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for group, (p0_args, p1_args, center_fraction, spread) in dynamics.items():
        matrices[group] = (
            _banded_shift_matrix(num_states, up1=p0_args[0], up2=p0_args[1], down1=p0_args[2], down2=p0_args[3]),
            _banded_shift_matrix(num_states, up1=p1_args[0], up2=p1_args[1], down1=p1_args[2], down2=p1_args[3]),
            _centered_belief(
                num_states,
                center=max(1, int(round(center_fraction * (num_states - 1)))),
                spread=spread,
            ),
        )

    P0 = np.empty((num_arms, num_states, num_states), dtype=float)
    P1 = np.empty_like(P0)
    initial_beliefs = np.empty((num_arms, num_states), dtype=float)
    for arm in range(num_arms):
        if arm < project_arms:
            group = "project"
        elif arm < project_arms + rollout_arms:
            group = "rollout"
        elif arm < cash_start:
            group = "vfa"
        else:
            group = "cash"
        P0_base, P1_base, center = matrices[group]
        P0[arm] = _regularized_probability_matrix(P0_base, rng, mix=0.008, jitter=0.014)
        P1[arm] = _regularized_probability_matrix(P1_base, rng, mix=0.010, jitter=0.016)
        initial_beliefs[arm] = _noisy_belief(center, rng, strength=600.0)

    R = np.broadcast_to(shared_R, (num_arms, num_states, num_states)).copy()
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
        name=f"{case}_tuned_shared_mc_d{num_states}_N{num_arms}_K{budget}_T{horizon}",
    )


def _shared_reward_matrix(
    reward_row: np.ndarray,
    rng: np.random.Generator,
    *,
    obs_amplitude: float,
    interaction_scale: float,
) -> np.ndarray:
    num_states = reward_row.size
    obs_profile = obs_amplitude * np.sin(np.linspace(0.0, 2.0 * np.pi, num_states, endpoint=False))
    interaction = rng.normal(0.0, interaction_scale, size=(num_states, num_states))
    return np.maximum(reward_row[:, None] + obs_profile[None, :] + interaction, 0.0)


def run_case(
    instance: MultiStateBatchInstance,
    *,
    episodes: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    results: dict[str, object] = {}
    for policy_index, policy_name in enumerate(POLICY_ORDER):
        traces: list[list[float]] = []
        unique_arms: list[int] = []
        decision_seconds = 0.0
        for episode in range(episodes):
            initial_states = _sample_initial_states(instance, seed=_seed_int(seed, 7001, episode))
            simulator = MultiStateBatchSimulator(
                instance,
                seed=_seed_int(seed, 9001, episode),
                initial_states=initial_states,
            )
            policy = _build_policy(policy_name, _seed_int(seed, 11003, policy_index, episode))
            trace = []
            arms = []
            for _ in range(instance.horizon):
                started = time.perf_counter()
                active = policy(simulator.beliefs.copy(), instance, simulator.time, simulator.rng)
                decision_seconds += time.perf_counter() - started
                arms.append(int(active[0]))
                step = simulator.step(active)
                trace.append(step.average_reward)
            traces.append(trace)
            unique_arms.append(len(set(arms)))
        trace_array = np.asarray(traces, dtype=float)
        results[policy_name] = {
            "traces": trace_array.tolist(),
            "mean_trace": np.mean(trace_array, axis=0).tolist(),
            "final_values": trace_array[:, -1].tolist(),
            "mean_final": float(np.mean(trace_array[:, -1])),
            "std_final": float(np.std(trace_array[:, -1], ddof=1)) if episodes > 1 else 0.0,
            "mean_unique_active_arms": float(np.mean(unique_arms)),
            "decision_seconds": float(decision_seconds),
            "mean_decision_ms": float(1000.0 * decision_seconds / max(1, episodes * instance.horizon)),
        }

    final = {policy: float(results[policy]["mean_final"]) for policy in POLICY_ORDER}  # type: ignore[index]
    best_baseline_name = max((policy for policy in POLICY_ORDER if policy != "our_index"), key=lambda p: final[p])
    best_baseline = final[best_baseline_name]
    metrics = {
        "instance": instance.name,
        "episodes": episodes,
        "final": final,
        "best_baseline": best_baseline_name,
        "final_gap_pct_vs_best_baseline": float(
            100.0 * (final["our_index"] - best_baseline) / max(1e-12, abs(best_baseline))
        ),
        "mean_unique_active_arms": {
            policy: float(results[policy]["mean_unique_active_arms"]) for policy in POLICY_ORDER  # type: ignore[index]
        },
        "mean_decision_ms": {
            policy: float(results[policy]["mean_decision_ms"]) for policy in POLICY_ORDER  # type: ignore[index]
        },
    }
    return results, metrics


def _build_policy(policy_name: str, seed: int):
    if policy_name == "our_index":
        return OurHybridTailIndexPolicy(
            prefix_depth=1,
            tail_horizon=24,
            tail_mode="commitment_envelope",
            resource_price=0.0,
            candidate_count=4,
            cache_precision=7,
        )
    if policy_name == "rollout_myopic":
        return ExpectedRolloutMyopicMultiStatePolicy(rollout_horizon=2, candidate_actions=4)
    if policy_name == "simple_vfa":
        return SimpleVFAMultiStatePolicy(
            immediate_weight=1.0,
            continuation_weight=0.04,
            quality_weight=0.02,
            information_weight=0.0,
            drift_weight=0.0,
        )
    if policy_name == "myopic":
        return MyopicMultiStatePolicy()
    raise ValueError(f"unknown policy {policy_name}")


def _sample_initial_states(instance: MultiStateBatchInstance, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [rng.choice(instance.num_states, p=belief) for belief in instance.initial_beliefs],
        dtype=int,
    )


def _seed_int(*parts: int) -> int:
    sequence = np.random.SeedSequence([int(part) & 0xFFFFFFFF for part in parts])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def plot_case(
    case: str,
    results: dict[str, object],
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    x = np.arange(1, len(results["our_index"]["mean_trace"]) + 1)  # type: ignore[index]
    plt.figure(figsize=(9.0, 5.4))
    for policy in POLICY_ORDER:
        policy_results = results[policy]  # type: ignore[index]
        traces = np.asarray(policy_results["traces"], dtype=float)  # type: ignore[index]
        mean_trace = np.asarray(policy_results["mean_trace"], dtype=float)  # type: ignore[index]
        color = POLICY_COLORS[policy]
        for trace in traces:
            plt.plot(x, trace, color=color, alpha=0.13, linewidth=0.8)
        label = (
            f"{POLICY_LABELS[policy]} "
            f"({metrics['mean_decision_ms'][policy]:.2f} ms/step)"  # type: ignore[index]
        )
        plt.plot(
            x,
            mean_trace,
            color=color,
            linestyle=POLICY_STYLES[policy],
            linewidth=2.4,
            label=label,
        )
    plt.xlabel("Time horizon prefix h")
    plt.ylabel("Average per-step cumulative reward")
    plt.title(
        f"{case}: sampled MC, N=100, d=8, K=1, beta=0.99, "
        f"gap={metrics['final_gap_pct_vs_best_baseline']:.1f}%"
    )
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
