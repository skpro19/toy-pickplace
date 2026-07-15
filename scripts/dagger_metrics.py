"""Aggregate and plot threshold-based DAgger collection metrics."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASK_METRIC_NAMES = (
    "grasped",
    "lifted",
    "tray_reached",
    "lowered_to_tray",
    "released_over_tray",
    "placement_success",
)
QUANTILES = {
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}


def _control_segment_lengths(*, execute_expert: np.ndarray) -> list[int]:
    padded = np.pad(execute_expert.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return (ends - starts).tolist()


def _quantiles(*, values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, quantile))
        for name, quantile in QUANTILES.items()
    }


def _load_dagger_episodes(*, dagger_dir: Path) -> list[dict[str, object]]:
    paths = sorted(dagger_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No DAgger npz files found in {dagger_dir}")

    episodes = []
    for path in paths:
        with np.load(path) as data:
            required = {
                "arm_disagreement",
                "execute_expert",
                "gripper_disagreement",
                "intervention_mode",
                "intervention_threshold",
                "intervention_steps",
                *TASK_METRIC_NAMES,
            }
            missing = required - set(data.files)
            if missing:
                raise KeyError(f"{path} is missing keys: {sorted(missing)}")

            arm_disagreement = np.asarray(
                data["arm_disagreement"], dtype=np.float64
            ).copy()
            execute_expert = np.asarray(data["execute_expert"], dtype=np.bool_).copy()
            gripper_disagreement = np.asarray(
                data["gripper_disagreement"], dtype=np.bool_
            ).copy()
            if not (
                arm_disagreement.ndim
                == execute_expert.ndim
                == gripper_disagreement.ndim
                == 1
            ):
                raise ValueError(f"{path} DAgger metric arrays must be one-dimensional")
            if not (
                len(arm_disagreement)
                == len(execute_expert)
                == len(gripper_disagreement)
            ):
                raise ValueError(f"{path} DAgger metric arrays have mismatched lengths")
            if len(arm_disagreement) == 0:
                raise ValueError(f"{path} contains no DAgger steps")

            episodes.append(
                {
                    "arm_disagreement": arm_disagreement,
                    "execute_expert": execute_expert,
                    "gripper_disagreement": gripper_disagreement,
                    "intervention_mode": str(data["intervention_mode"].item()),
                    "intervention_threshold": float(
                        data["intervention_threshold"].item()
                    ),
                    "intervention_steps": int(data["intervention_steps"].item()),
                    "task_metrics": {
                        name: bool(data[name].item()) for name in TASK_METRIC_NAMES
                    },
                }
            )
    return episodes


def summarize_dagger_round(
    *, dagger_dir: Path, round_index: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    episodes = _load_dagger_episodes(dagger_dir=dagger_dir)
    modes = {str(episode["intervention_mode"]) for episode in episodes}
    thresholds = {
        float(episode["intervention_threshold"]) for episode in episodes
    }
    intervention_steps = {int(episode["intervention_steps"]) for episode in episodes}
    if modes != {"threshold"}:
        raise ValueError(f"Expected threshold DAgger data, found modes: {sorted(modes)}")
    if len(thresholds) != 1 or len(intervention_steps) != 1:
        raise ValueError("DAgger intervention configuration differs between episodes")

    threshold = thresholds.pop()
    all_arm_disagreement = np.concatenate(
        [np.asarray(episode["arm_disagreement"]) for episode in episodes]
    )
    all_execute_expert = np.concatenate(
        [np.asarray(episode["execute_expert"]) for episode in episodes]
    )
    all_gripper_disagreement = np.concatenate(
        [np.asarray(episode["gripper_disagreement"]) for episode in episodes]
    )
    episode_expert_fractions = np.asarray(
        [
            np.mean(np.asarray(episode["execute_expert"], dtype=np.float64))
            for episode in episodes
        ]
    )
    segment_lengths = [
        length
        for episode in episodes
        for length in _control_segment_lengths(
            execute_expert=np.asarray(episode["execute_expert"])
        )
    ]
    segment_array = np.asarray(segment_lengths, dtype=np.float64)

    summary: dict[str, object] = {
        "round": round_index,
        "episodes": len(episodes),
        "steps": int(len(all_arm_disagreement)),
        "intervention_threshold": threshold,
        "intervention_steps": intervention_steps.pop(),
        "expert_action_fraction": float(np.mean(all_execute_expert)),
        "episode_expert_action_fraction": {
            "mean": float(np.mean(episode_expert_fractions)),
            "p50": float(np.quantile(episode_expert_fractions, 0.50)),
            "p95": float(np.quantile(episode_expert_fractions, 0.95)),
        },
        "threshold_exceedance_fraction": float(
            np.mean(all_arm_disagreement > threshold)
        ),
        "intervention_trigger_fraction": float(
            np.mean(
                (all_arm_disagreement > threshold) | all_gripper_disagreement
            )
        ),
        "arm_disagreement_quantiles": _quantiles(values=all_arm_disagreement),
        "gripper_disagreement_fraction": float(
            np.mean(all_gripper_disagreement)
        ),
        "expert_control_segments": len(segment_lengths),
        "expert_segment_steps": {
            "mean": float(np.mean(segment_array)) if len(segment_array) else 0.0,
            "p95": (
                float(np.quantile(segment_array, 0.95)) if len(segment_array) else 0.0
            ),
            "max": int(np.max(segment_array)) if len(segment_array) else 0,
        },
        "task_success_rates": {
            name: float(
                np.mean(
                    [bool(episode["task_metrics"][name]) for episode in episodes]
                )
            )
            for name in TASK_METRIC_NAMES
        },
    }
    return summary, episodes


def write_dagger_metrics(*, metrics_path: Path, metrics: dict[str, object]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


def plot_dagger_round(
    *,
    episodes: list[dict[str, object]],
    summary: dict[str, object],
    save_path: Path,
) -> None:
    threshold = float(summary["intervention_threshold"])
    all_disagreement = np.concatenate(
        [np.asarray(episode["arm_disagreement"]) for episode in episodes]
    )
    segment_lengths = [
        length
        for episode in episodes
        for length in _control_segment_lengths(
            execute_expert=np.asarray(episode["execute_expert"])
        )
    ]
    max_steps = max(len(np.asarray(episode["arm_disagreement"])) for episode in episodes)
    disagreement_by_step = np.full((len(episodes), max_steps), np.nan)
    expert_by_step = np.full((len(episodes), max_steps), np.nan)
    for index, episode in enumerate(episodes):
        disagreement = np.asarray(episode["arm_disagreement"])
        execute_expert = np.asarray(episode["execute_expert"])
        disagreement_by_step[index, : len(disagreement)] = disagreement
        expert_by_step[index, : len(execute_expert)] = execute_expert

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    histogram_ax, timeline_ax, segment_ax, task_ax = axes.flat

    histogram_ax.hist(all_disagreement, bins=60, color="tab:blue", alpha=0.8)
    histogram_ax.axvline(threshold, color="tab:red", linestyle="--", label="threshold")
    histogram_ax.set(title="Arm disagreement", xlabel="L2 disagreement", ylabel="Steps")
    histogram_ax.legend()

    steps = np.arange(max_steps)
    timeline_ax.plot(
        steps,
        np.nanmean(disagreement_by_step, axis=0),
        color="tab:blue",
        label="mean disagreement",
    )
    timeline_ax.axhline(threshold, color="tab:red", linestyle="--", label="threshold")
    timeline_ax.set(xlabel="Step", ylabel="L2 disagreement", title="Collection timeline")
    expert_ax = timeline_ax.twinx()
    expert_ax.plot(
        steps,
        np.nanmean(expert_by_step, axis=0),
        color="tab:orange",
        alpha=0.8,
        label="expert fraction",
    )
    expert_ax.set_ylabel("Expert fraction")
    lines = timeline_ax.lines + expert_ax.lines
    timeline_ax.legend(lines, [line.get_label() for line in lines], fontsize=8)

    if segment_lengths:
        segment_ax.hist(segment_lengths, bins=min(30, len(segment_lengths)), color="tab:orange")
    else:
        segment_ax.text(0.5, 0.5, "No expert-control segments", ha="center", va="center")
    segment_ax.set(
        title=f"Expert-control segments (n={len(segment_lengths)})",
        xlabel="Duration (steps)",
        ylabel="Segments",
    )

    task_rates = summary["task_success_rates"]
    task_names = list(TASK_METRIC_NAMES)
    task_ax.bar(
        range(len(task_names)),
        [float(task_rates[name]) for name in task_names],
        color="tab:green",
    )
    task_ax.set_xticks(range(len(task_names)))
    task_ax.set_xticklabels(task_names, rotation=30, ha="right", fontsize=8)
    task_ax.set_ylim(0.0, 1.0)
    task_ax.set(title="Task outcomes", ylabel="Episode rate")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f'DAgger round {int(summary["round"]):03d} | '
        f'expert={float(summary["expert_action_fraction"]):.1%}'
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_dagger_round_trends(
    *, round_summaries: list[dict[str, object]], save_path: Path
) -> None:
    if not round_summaries:
        raise ValueError("At least one DAgger round summary is required")

    rounds = [int(summary["round"]) for summary in round_summaries]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    disagreement_ax, rates_ax, segments_ax, duration_ax = axes.flat

    for quantile in ("p50", "p90", "p95", "p99"):
        disagreement_ax.plot(
            rounds,
            [
                float(summary["arm_disagreement_quantiles"][quantile])
                for summary in round_summaries
            ],
            marker="o",
            label=quantile,
        )
    disagreement_ax.plot(
        rounds,
        [float(summary["intervention_threshold"]) for summary in round_summaries],
        color="black",
        linestyle="--",
        label="threshold",
    )
    disagreement_ax.set(title="Arm disagreement", ylabel="L2 disagreement")
    disagreement_ax.legend(fontsize=8)

    rate_series = {
        "expert action": "expert_action_fraction",
        "intervention trigger": "intervention_trigger_fraction",
        "above threshold": "threshold_exceedance_fraction",
        "gripper disagreement": "gripper_disagreement_fraction",
    }
    for label, key in rate_series.items():
        rates_ax.plot(
            rounds,
            [float(summary[key]) for summary in round_summaries],
            marker="o",
            label=label,
        )
    rates_ax.plot(
        rounds,
        [
            float(summary["task_success_rates"]["placement_success"])
            for summary in round_summaries
        ],
        marker="o",
        label="placement success",
    )
    rates_ax.set(title="Collection rates", ylabel="Fraction", ylim=(0.0, 1.0))
    rates_ax.legend(fontsize=8)

    segments_ax.plot(
        rounds,
        [int(summary["expert_control_segments"]) for summary in round_summaries],
        marker="o",
        color="tab:orange",
    )
    segments_ax.set(title="Expert-control segments", xlabel="Round", ylabel="Count")

    for statistic in ("mean", "p95", "max"):
        duration_ax.plot(
            rounds,
            [
                float(summary["expert_segment_steps"][statistic])
                for summary in round_summaries
            ],
            marker="o",
            label=statistic,
        )
    duration_ax.set(title="Expert segment duration", xlabel="Round", ylabel="Steps")
    duration_ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xticks(rounds)
        ax.grid(True, alpha=0.25)
    fig.suptitle("DAgger collection trends")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
