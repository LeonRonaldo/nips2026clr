from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POLICY_ORDER = ["our_index", "rollout_myopic", "myopic"]
POLICY_LABELS = {
    "our_index": "Ours (tail index)",
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
        description="Export averaged MC paths from saved tuned shared-R/shared-E traces."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("exp1/outputs/exp1"),
        help="Directory containing traces.json and metrics.json.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("exp1/csv"),
        help="Output directory for average MC path CSV files.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("exp1/figures"),
        help="Output directory for average MC path figures.",
    )
    parser.add_argument(
        "--show-episode-traces",
        action="store_true",
        help="Overlay individual MC episode traces behind the mean curves.",
    )
    parser.add_argument(
        "--show-sem-band",
        action="store_true",
        help="Show a 95% standard-error band around each mean curve.",
    )
    args = parser.parse_args()

    traces_path = args.input_dir / "traces.json"
    metrics_path = args.input_dir / "metrics.json"
    traces = json.loads(traces_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, Any]] = []
    for case in sorted(traces):
        arrays = {
            policy: np.asarray(traces[case][policy]["traces"], dtype=float)
            for policy in POLICY_ORDER
        }
        case_rows, case_long_rows = average_case_rows(case, arrays, metrics[case])
        long_rows.extend(case_long_rows)

        case_csv = args.csv_dir / f"{case}_average_mc_path.csv"
        write_csv(case_csv, case_rows)
        plot_case(
            case=case,
            arrays=arrays,
            metrics=metrics[case],
            output_path=args.figure_dir / f"{case}_average_mc_path.png",
            show_episode_traces=args.show_episode_traces,
            show_sem_band=args.show_sem_band,
        )
        print(f"WROTE_CSV: {case_csv}")
        print(f"WROTE_FIGURE: {args.figure_dir / f'{case}_average_mc_path.png'}")

    long_csv = args.csv_dir / "all_cases_average_mc_path_long.csv"
    write_csv(long_csv, long_rows)
    print(f"WROTE_LONG_CSV: {long_csv}")


def average_case_rows(
    case: str,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    horizon = next(iter(arrays.values())).shape[1]
    episodes = next(iter(arrays.values())).shape[0]
    case_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    means = {policy: arrays[policy].mean(axis=0) for policy in POLICY_ORDER}
    stds = {
        policy: arrays[policy].std(axis=0, ddof=1) if episodes > 1 else np.zeros(horizon)
        for policy in POLICY_ORDER
    }
    sems = {policy: stds[policy] / np.sqrt(max(1, episodes)) for policy in POLICY_ORDER}

    for t in range(horizon):
        row: dict[str, Any] = {"case": case, "horizon": t + 1, "episodes": episodes}
        for policy in POLICY_ORDER:
            row[f"{policy}_mean"] = float(means[policy][t])
            row[f"{policy}_std"] = float(stds[policy][t])
            row[f"{policy}_sem"] = float(sems[policy][t])
            long_rows.append(
                {
                    "case": case,
                    "horizon": t + 1,
                    "method": policy,
                    "method_label": POLICY_LABELS[policy],
                    "episodes": episodes,
                    "mean_avg_per_step_cumulative_reward": float(means[policy][t]),
                    "std_avg_per_step_cumulative_reward": float(stds[policy][t]),
                    "sem_avg_per_step_cumulative_reward": float(sems[policy][t]),
                    "mean_decision_ms": float(metrics["mean_decision_ms"][policy]),
                    "final_gap_pct_vs_best_baseline": float(
                        metrics["final_gap_pct_vs_best_baseline"]
                    ),
                }
            )
        case_rows.append(row)
    return case_rows, long_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_case(
    *,
    case: str,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
    output_path: Path,
    show_episode_traces: bool,
    show_sem_band: bool,
) -> None:
    x = np.arange(1, next(iter(arrays.values())).shape[1] + 1)
    plt.figure(figsize=(9.0, 5.4))
    for policy in POLICY_ORDER:
        traces = arrays[policy]
        mean_trace = traces.mean(axis=0)
        std_trace = traces.std(axis=0, ddof=1) if traces.shape[0] > 1 else np.zeros_like(mean_trace)
        sem_trace = std_trace / np.sqrt(max(1, traces.shape[0]))
        color = POLICY_COLORS[policy]
        if show_episode_traces:
            for trace in traces:
                plt.plot(x, trace, color=color, alpha=0.11, linewidth=0.75)
        if show_sem_band:
            plt.fill_between(
                x,
                mean_trace - 1.96 * sem_trace,
                mean_trace + 1.96 * sem_trace,
                color=color,
                alpha=0.10,
                linewidth=0.0,
            )
        label = (
            f"{POLICY_LABELS[policy]} "
            f"({metrics['mean_decision_ms'][policy]:.2f} ms/step)"
        )
        plt.plot(
            x,
            mean_trace,
            color=color,
            linestyle=POLICY_STYLES[policy],
            linewidth=2.35,
            label=label,
        )

    plt.xlabel("Time horizon prefix h")
    plt.ylabel("Average per-step cumulative reward")
    plt.title(
        f"{case}: mean MC path, N=100, d=8, K=1, beta=0.99, "
        f"episodes={next(iter(arrays.values())).shape[0]}, "
        f"gap={metrics['final_gap_pct_vs_best_baseline']:.1f}%"
    )
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
