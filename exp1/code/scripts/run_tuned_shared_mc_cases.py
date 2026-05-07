from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from canonical_rmab.multistate_large_scale import (
    MultiStateBatchInstance,
    MultiStateBatchSimulator,
    MyopicMultiStatePolicy,
    OurHybridTailIndexPolicy,
    RolloutMyopicMultiStatePolicy,
)


POLICY_ORDER = ["our_index", "rollout_myopic", "myopic"]
POLICY_LABELS = {
    "our_index": "Ours (full tail index)",
    "rollout_myopic": "Rollout-myopic",
    "myopic": "Myopic",
}
POLICY_COLORS = {
    "our_index": "#1f77b4",
    "rollout_myopic": "#d62728",
    "myopic": "#7f7f7f",
}
POLICY_STYLES = {
    "our_index": "-",
    "rollout_myopic": "--",
    "myopic": ":",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sampled-MC evaluation for the anonymous three-case instances."
    )
    parser.add_argument("--num-arms", type=int, default=100)
    parser.add_argument("--num-states", type=int, default=8)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=Path("../json"),
        help="Directory containing case*_100arm_matrices.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/exp1"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()
    all_results: dict[str, object] = {}
    all_metrics: dict[str, object] = {}
    for case in ("case1", "case2", "case3"):
        instance = load_instance(
            case=case,
            matrix_dir=args.matrix_dir,
            num_arms=args.num_arms,
            num_states=args.num_states,
            budget=args.budget,
            horizon=args.horizon,
            discount=args.discount,
        )
        case_results, case_metrics = run_case(instance, episodes=args.episodes, rng=rng)
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
            f"myopic={case_metrics['final']['myopic']:.4f}, "
            f"gap={case_metrics['final_gap_pct_vs_best_baseline']:.2f}%"
        )

    (args.output_dir / "traces.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"OUTPUT_DIR: {args.output_dir}")


def load_instance(
    *,
    case: str,
    matrix_dir: Path,
    num_arms: int,
    num_states: int,
    budget: int,
    horizon: int,
    discount: float,
) -> MultiStateBatchInstance:
    path = matrix_dir / f"{case}_100arm_matrices.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    transitions: dict[str, Any] = data["transition_matrices"]
    observations: dict[str, Any] = data["observation_error_matrices"]
    rewards: dict[str, Any] = data["reward_matrices"]

    p0 = np.asarray(transitions["P0_passive"], dtype=float)
    p1 = np.asarray(transitions["P1_active"], dtype=float)
    e1 = np.asarray(observations["E1_active"], dtype=float)
    e0_raw = observations.get("E0_passive")
    e0 = None if e0_raw is None else np.asarray(e0_raw, dtype=float)
    r = np.asarray(rewards["R"], dtype=float)
    initial_beliefs = np.asarray(data["initial_beliefs"], dtype=float)

    if p0.shape != (num_arms, num_states, num_states):
        raise ValueError(f"{case} P0 shape {p0.shape} does not match N={num_arms}, d={num_states}.")

    return MultiStateBatchInstance(
        P0=p0,
        P1=p1,
        E0=e0,
        E1=e1,
        R=r,
        initial_beliefs=initial_beliefs,
        budget=budget,
        horizon=horizon,
        discount=discount,
        name=f"{case}_N{num_arms}_d{num_states}_K{budget}_T{horizon}",
    )


def run_case(
    instance: MultiStateBatchInstance,
    *,
    episodes: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, object]]:
    results: dict[str, object] = {}
    for policy_index, policy_name in enumerate(POLICY_ORDER):
        traces: list[list[float]] = []
        unique_arms: list[int] = []
        decision_seconds = 0.0
        for _ in range(episodes):
            initial_states = _sample_initial_states(instance, rng)
            simulator = MultiStateBatchSimulator(instance, _random_int(rng), initial_states=initial_states)
            policy = _build_policy(policy_name, instance, rng)
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


def _build_policy(
    policy_name: str,
    instance: MultiStateBatchInstance,
    rng: np.random.Generator,
):
    if policy_name == "our_index":
        return OurHybridTailIndexPolicy(
            prefix_depth=1,
            tail_horizon=instance.horizon,
            tail_mode="commitment_envelope",
            resource_price=0.0,
            candidate_count=None,
            cache_precision=7,
        )
    if policy_name == "rollout_myopic":
        return RolloutMyopicMultiStatePolicy(4, 1000, None, _random_int(rng))
    if policy_name == "myopic":
        return MyopicMultiStatePolicy()
    raise ValueError(f"unknown policy {policy_name}")


def _sample_initial_states(instance: MultiStateBatchInstance, rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [rng.choice(instance.num_states, p=belief) for belief in instance.initial_beliefs],
        dtype=int,
    )


def _random_int(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**32 - 1))


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
