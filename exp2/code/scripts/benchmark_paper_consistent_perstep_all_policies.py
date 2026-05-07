from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_tuned_shared_mc_cases import make_tuned_shared_instance  # noqa: E402
from canonical_rmab.multistate_large_scale import (  # noqa: E402
    MultiStateBatchSimulator,
    MyopicMultiStatePolicy,
    OurHybridTailIndexPolicy,
    RolloutMyopicMultiStatePolicy,
    SimpleVFAMultiStatePolicy,
)


CASES = ("case1", "case2", "case3")
POLICY_ORDER = ["ours_full_tail_all_arms", "rollout_depth4_mc1000", "simple_vfa", "myopic"]
POLICY_LABELS = {
    "ours_full_tail_all_arms": "Ours full-tail all-arms",
    "rollout_depth4_mc1000": "Rollout depth 4 MC1000",
    "simple_vfa": "VFA",
    "myopic": "Myopic",
}
POLICY_COLORS = {
    "ours_full_tail_all_arms": "#1f77b4",
    "rollout_depth4_mc1000": "#d62728",
    "simple_vfa": "#2ca02c",
    "myopic": "#7f7f7f",
}
POLICY_STYLES = {
    "ours_full_tail_all_arms": "-",
    "rollout_depth4_mc1000": "--",
    "simple_vfa": "-.",
    "myopic": ":",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-step per-step runtime benchmark for all policies on the paper-consistent grid."
    )
    parser.add_argument("--num-states", type=int, default=8)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fixed-n", type=int, default=100)
    parser.add_argument("--horizons", type=str, default="5,10,20,40,50,80,100,150,200")
    parser.add_argument("--fixed-horizon", type=int, default=100)
    parser.add_argument("--arm-counts", type=str, default="10,20,30,40,50,80,100")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nipsexp/data/final_runtime_paper_consistent_simple_all_policies"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("nipsexp/figures/final_runtime_paper_consistent_simple_all_policies"),
    )
    args = parser.parse_args()

    horizons = [int(value) for value in args.horizons.split(",") if value.strip()]
    arm_counts = [int(value) for value in args.arm_counts.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        for case in CASES:
            for policy_name in POLICY_ORDER:
                for repeat in range(args.repeats):
                    raw_rows.append(
                        measure_once(
                            sweep="fixed_N_vary_T",
                            case=case,
                            policy_name=policy_name,
                            num_arms=args.fixed_n,
                            horizon=horizon,
                            num_states=args.num_states,
                            budget=args.budget,
                            discount=args.discount,
                            seed=seed_int(args.seed, horizon, repeat, policy_index(policy_name)),
                            repeat=repeat,
                        )
                    )
    for num_arms in arm_counts:
        for case in CASES:
            for policy_name in POLICY_ORDER:
                for repeat in range(args.repeats):
                    raw_rows.append(
                        measure_once(
                            sweep="fixed_T_vary_N",
                            case=case,
                            policy_name=policy_name,
                            num_arms=num_arms,
                            horizon=args.fixed_horizon,
                            num_states=args.num_states,
                            budget=args.budget,
                            discount=args.discount,
                            seed=seed_int(args.seed, num_arms, repeat, policy_index(policy_name)),
                            repeat=repeat,
                        )
                    )

    summary_rows = aggregate(raw_rows)
    raw_path = args.output_dir / "paper_consistent_perstep_all_policies_raw.csv"
    summary_path = args.output_dir / "paper_consistent_perstep_all_policies_summary.csv"
    write_csv(raw_path, raw_rows)
    write_csv(summary_path, summary_rows)
    write_csv(args.figure_dir / summary_path.name, summary_rows)
    write_csv(args.figure_dir / raw_path.name, raw_rows)

    metadata = {
        "measurement": "single cold action-selection call at t=0 from initial public belief",
        "policies": {
            "ours_full_tail_all_arms": (
                "OurHybridTailIndexPolicy(prefix_depth=1, tail_horizon=instance.horizon, "
                "candidate_count=None, tail_mode=commitment_envelope, resource_price=0)."
            ),
            "rollout_depth4_mc1000": (
                "RolloutMyopicMultiStatePolicy(rollout_horizon=4, rollout_samples=1000, "
                "candidate_actions=4)."
            ),
            "simple_vfa": "SimpleVFAMultiStatePolicy with the tuned weights used in the main figures.",
            "myopic": "MyopicMultiStatePolicy.",
        },
        "fixed_N_vary_T": {"N": args.fixed_n, "horizons": horizons},
        "fixed_T_vary_N": {"horizon": args.fixed_horizon, "N_values": arm_counts},
        "cases": list(CASES),
        "repeats": args.repeats,
    }
    metadata_path = args.output_dir / "paper_consistent_perstep_all_policies_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.figure_dir / metadata_path.name).write_text(metadata_path.read_text(), encoding="utf-8")

    plot(summary_rows, args.figure_dir)
    print(f"RAW_CSV: {raw_path}")
    print(f"SUMMARY_CSV: {summary_path}")
    print(f"FIGURE_DIR: {args.figure_dir}")


def measure_once(
    *,
    sweep: str,
    case: str,
    policy_name: str,
    num_arms: int,
    horizon: int,
    num_states: int,
    budget: int,
    discount: float,
    seed: int,
    repeat: int,
) -> dict[str, Any]:
    instance = make_tuned_shared_instance(
        case=case,
        num_arms=num_arms,
        num_states=num_states,
        budget=budget,
        horizon=horizon,
        discount=discount,
        seed=seed,
    )
    simulator = MultiStateBatchSimulator(instance, seed=seed)
    policy = build_policy(policy_name, instance, seed)
    started = time.perf_counter()
    action = policy(simulator.beliefs.copy(), instance, simulator.time, simulator.rng)
    decision_ms = 1000.0 * (time.perf_counter() - started)
    return {
        "sweep": sweep,
        "case": case,
        "N": num_arms,
        "d": num_states,
        "K": budget,
        "horizon": horizon,
        "repeat": repeat,
        "policy": policy_name,
        "policy_label": POLICY_LABELS[policy_name],
        "decision_ms": float(decision_ms),
        "selected_arm": int(action[0]),
    }


def build_policy(policy_name: str, instance, seed: int):
    if policy_name == "ours_full_tail_all_arms":
        return OurHybridTailIndexPolicy(
            prefix_depth=1,
            tail_horizon=instance.horizon,
            tail_mode="commitment_envelope",
            resource_price=0.0,
            candidate_count=None,
            cache_precision=7,
        )
    if policy_name == "rollout_depth4_mc1000":
        return RolloutMyopicMultiStatePolicy(
            rollout_horizon=4,
            rollout_samples=1000,
            candidate_actions=4,
            seed=seed,
        )
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


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int, str], list[float]] = {}
    for row in rows:
        key = (str(row["sweep"]), int(row["N"]), int(row["horizon"]), str(row["policy"]))
        groups.setdefault(key, []).append(float(row["decision_ms"]))
    output: list[dict[str, Any]] = []
    for (sweep, num_arms, horizon, policy), values in sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2], policy_index(item[0][3])),
    ):
        arr = np.asarray(values, dtype=float)
        output.append(
            {
                "sweep": sweep,
                "N": num_arms,
                "horizon": horizon,
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "mean_decision_ms": float(np.mean(arr)),
                "std_decision_ms": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "samples": len(values),
            }
        )
    return output


def plot(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    plot_single(
        rows,
        sweep="fixed_N_vary_T",
        x_key="horizon",
        xlabel="Horizon T",
        title="One-step per-step runtime vs T, N=100",
        output_path=figure_dir / "paper_consistent_all_policies_perstep_vs_T_N100.png",
    )
    plot_single(
        rows,
        sweep="fixed_T_vary_N",
        x_key="N",
        xlabel="Number of arms N",
        title="One-step per-step runtime vs N, T=100",
        output_path=figure_dir / "paper_consistent_all_policies_perstep_vs_N_T100.png",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    draw_panel(
        axes[0],
        rows,
        sweep="fixed_N_vary_T",
        x_key="horizon",
        xlabel="Horizon T",
        title="Fixed N=100",
        show_legend=False,
    )
    draw_panel(
        axes[1],
        rows,
        sweep="fixed_T_vary_N",
        x_key="N",
        xlabel="Number of arms N",
        title="Fixed T=100",
        show_legend=False,
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("One-step runtime: full-tail ours and baselines", y=0.91, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(figure_dir / "paper_consistent_all_policies_perstep_combined.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_single(
    rows: list[dict[str, Any]],
    *,
    sweep: str,
    x_key: str,
    xlabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    draw_panel(ax, rows, sweep=sweep, x_key=x_key, xlabel=xlabel, title=title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_panel(
    ax,
    rows: list[dict[str, Any]],
    *,
    sweep: str,
    x_key: str,
    xlabel: str,
    title: str,
    show_legend: bool = True,
) -> None:
    for policy in POLICY_ORDER:
        policy_rows = sorted(
            [row for row in rows if row["sweep"] == sweep and row["policy"] == policy],
            key=lambda row: int(row[x_key]),
        )
        ax.plot(
            [int(row[x_key]) for row in policy_rows],
            [float(row["mean_decision_ms"]) for row in policy_rows],
            marker="o",
            linewidth=2.25,
            color=POLICY_COLORS[policy],
            linestyle=POLICY_STYLES[policy],
            label=POLICY_LABELS[policy],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("One-step decision runtime (ms)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    if show_legend:
        ax.legend(frameon=False, fontsize=8.4)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def policy_index(policy_name: str) -> int:
    return POLICY_ORDER.index(policy_name)


def seed_int(*parts: int) -> int:
    sequence = np.random.SeedSequence([int(part) & 0xFFFFFFFF for part in parts])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


if __name__ == "__main__":
    main()
