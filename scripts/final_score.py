"""Re-evaluate best.pt checkpoints from a completed flywheel run on a larger eval set.

Usage:
    uv run python scripts/final_score.py
    uv run python scripts/final_score.py --run-name run-019
    uv run python scripts/final_score.py --run-name run-019 --eval-episodes 200 --workers 8
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval import EvalSelectionMode, eval_selection_key, score_ckpt


SECTION_WIDTH = 72
FINAL_EVAL_SEED_COUNT = 5


def detect_latest_run(*, root: Path) -> str | None:
    pattern = re.compile(r"^run-(\d+)$")
    if not root.is_dir():
        return None
    indices = [
        int(m.group(1))
        for path in root.iterdir()
        if (m := pattern.match(path.name))
    ]
    if not indices:
        return None
    return f"run-{max(indices):03d}"


def make_final_eval_seeds(*, seed: int) -> list[int]:
    seed_sequence = np.random.SeedSequence(seed)
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(FINAL_EVAL_SEED_COUNT)
    ]


def aggregate_seed_results(*, seed_results: list[dict[str, object]]) -> dict[str, float | int]:
    if not seed_results:
        raise ValueError("Cannot aggregate an empty set of evaluation results")

    metric_keys = (
        "mean_score",
        "grasp_rate",
        "lift_rate",
        "tray_reach_rate",
        "lowered_to_tray_rate",
        "released_over_tray_rate",
        "placement_success_rate",
    )
    return {
        "eval_metric_version": int(seed_results[0]["eval_metric_version"]),
        **{
            metric_key: float(np.mean([result[metric_key] for result in seed_results]))
            for metric_key in metric_keys
        },
    }


def validate_complete_rounds(*, metrics_data: dict[str, object]) -> None:
    config = metrics_data.get("config", {})
    if not isinstance(config, dict) or "dagger_rounds" not in config:
        return

    expected_rounds = list(range(int(config["dagger_rounds"]) + 1))
    rounds = metrics_data.get("rounds", [])
    if not isinstance(rounds, list):
        raise ValueError("metrics rounds must be a list")
    actual_rounds = [int(item["round"]) for item in rounds]
    if actual_rounds != expected_rounds:
        raise ValueError(
            f"Incomplete flywheel metrics: expected rounds {expected_rounds}, "
            f"found {actual_rounds}"
        )


def resolve_final_eval_settings(
    *,
    config: dict[str, object],
    eval_episodes: int | None,
    final_eval_seed: int | None,
    workers: int | None,
    capture_hz: float | None,
) -> tuple[int, int, int, float]:
    resolved_episodes = (
        eval_episodes
        if eval_episodes is not None
        else int(config.get("final_eval_episodes", 100))
    )
    resolved_seed = (
        final_eval_seed
        if final_eval_seed is not None
        else int(config.get("final_eval_seed", 20260716))
    )
    resolved_workers = workers if workers is not None else int(
        config.get("final_eval_workers", config.get("workers", 6))
    )
    resolved_capture_hz = capture_hz if capture_hz is not None else float(
        config.get("final_eval_capture_hz", config.get("eval_capture_hz", 60.0))
    )

    if resolved_episodes < FINAL_EVAL_SEED_COUNT:
        raise ValueError(f"eval episodes must be at least {FINAL_EVAL_SEED_COUNT}")
    if resolved_episodes % FINAL_EVAL_SEED_COUNT != 0:
        raise ValueError(f"eval episodes must be divisible by {FINAL_EVAL_SEED_COUNT}")
    if resolved_seed < 0:
        raise ValueError("final eval seed must be non-negative")
    if resolved_workers < 1:
        raise ValueError("workers must be at least 1")
    if resolved_capture_hz <= 0.0:
        raise ValueError("capture_hz must be positive")

    return resolved_episodes, resolved_seed, resolved_workers, resolved_capture_hz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate best.pt checkpoints from a completed flywheel run"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Flywheel run name (e.g. run-019). Auto-detects latest if omitted.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=os.environ.get("FLYWHEEL_ARCH", ""),
        help="Architecture subdirectory (e.g. vision_mlp)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Total episodes per checkpoint (default: run config, otherwise 100)",
    )
    parser.add_argument(
        "--final-eval-seed",
        type=int,
        default=None,
        help="Root seed for five held-out seeds (default: run config)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from final_scores.json without evaluating checkpoints",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: final_eval_workers from run config)",
    )
    parser.add_argument(
        "--capture-hz",
        type=float,
        default=None,
        help="Policy inference rate (default: final_eval_capture_hz from run config)",
    )
    args = parser.parse_args()
    if args.eval_episodes is not None and args.eval_episodes < FINAL_EVAL_SEED_COUNT:
        parser.error(f"--eval-episodes must be at least {FINAL_EVAL_SEED_COUNT}")
    if (
        args.eval_episodes is not None
        and args.eval_episodes % FINAL_EVAL_SEED_COUNT != 0
    ):
        parser.error(
            f"--eval-episodes must be divisible by {FINAL_EVAL_SEED_COUNT}"
        )
    if args.final_eval_seed is not None and args.final_eval_seed < 0:
        parser.error("--final-eval-seed must be non-negative")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.capture_hz is not None and args.capture_hz <= 0.0:
        parser.error("--capture-hz must be positive")
    return args


def main() -> None:
    args = parse_args()

    results_root = Path("results/flywheel") / args.arch if args.arch else Path("results/flywheel")

    if args.run_name is None:
        detected = detect_latest_run(root=results_root)
        if detected is None:
            print("No flywheel runs found in results/flywheel/.")
            print("Specify --run-name to target a specific run.")
            return
        run_name = detected
        print(f"Auto-detected latest run: {run_name}")
    else:
        run_name = args.run_name

    run_results_dir = results_root / run_name
    metrics_path = run_results_dir / "metrics.json"
    if not metrics_path.is_file():
        print(f"Metrics file not found: {metrics_path}")
        return

    with open(metrics_path) as f:
        metrics_data = json.load(f)

    rounds = metrics_data.get("rounds", [])
    if not rounds:
        print("No rounds found in metrics.json")
        return

    config = metrics_data.get("config", {})
    eval_max_steps = metrics_data.get("eval_max_steps", 1400)
    original_episodes = metrics_data.get("eval_episodes", 25)
    eval_episodes, final_eval_seed, workers, capture_hz = resolve_final_eval_settings(
        config=config,
        eval_episodes=args.eval_episodes,
        final_eval_seed=args.final_eval_seed,
        workers=args.workers,
        capture_hz=args.capture_hz,
    )
    episodes_per_seed = eval_episodes // FINAL_EVAL_SEED_COUNT
    final_eval_seeds = make_final_eval_seeds(seed=final_eval_seed)
    selection_mode: EvalSelectionMode = metrics_data.get("eval_selection_mode", "mode-b")

    if args.plot_only:
        final_scores_path = run_results_dir / "final_scores.json"
        if not final_scores_path.is_file():
            print(f"Final scores file not found: {final_scores_path}")
            return
        final_scores_data = json.loads(final_scores_path.read_text())
        validate_complete_rounds(
            metrics_data={**metrics_data, "rounds": final_scores_data["rounds"]}
        )
        plot_comparison(
            final_rounds=final_scores_data["rounds"],
            run_name=run_name,
            save_dir=run_results_dir,
            eval_episodes=int(final_scores_data["eval_episodes"]),
            original_episodes=original_episodes,
        )
        return

    validate_complete_rounds(metrics_data=metrics_data)
    ckpt_root = Path("checkpoints/flywheel") / args.arch if args.arch else Path("checkpoints/flywheel")
    checkpoints: list[tuple[int, Path]] = []
    for round_data in rounds:
        round_idx = int(round_data["round"])
        ckpt_rel = round_data.get("best_checkpoint", "")
        ckpt_path = Path(ckpt_rel)
        if not ckpt_path.is_file():
            alt_path = ckpt_root / run_name / f"round-{round_idx:03d}" / "best.pt"
            if alt_path.is_file():
                ckpt_path = alt_path
            else:
                print(f"  WARNING: checkpoint not found for round {round_idx}: {ckpt_path}")
                continue
        checkpoints.append((round_idx, ckpt_path))

    if not checkpoints:
        raise RuntimeError("No valid checkpoints found to evaluate")
    if len(checkpoints) != len(rounds):
        raise RuntimeError(
            f"Expected {len(rounds)} checkpoints, found {len(checkpoints)}"
        )

    print(f"\nRun: {run_name}")
    print(
        f"Final eval: {FINAL_EVAL_SEED_COUNT} held-out seeds x "
        f"{episodes_per_seed} episodes x {eval_max_steps} steps, workers={workers}"
    )
    print(f"Held-out root seed: {final_eval_seed}")
    print(f"Capture rate: {capture_hz:g} Hz")
    print(f"Original eval: {original_episodes} episodes")
    print()

    final_rounds = []
    for round_idx, ckpt_path in checkpoints:
        original_score = rounds[round_idx].get("best_score", None) if round_idx < len(rounds) else None
        original_placement = rounds[round_idx].get("best_placement_success_rate", None) if round_idx < len(rounds) else None

        print(f"  Round {round_idx:03d}: {ckpt_path}")
        seed_results = []
        for seed_index, eval_seed in enumerate(final_eval_seeds, start=1):
            print(
                f"    Seed {seed_index}/{FINAL_EVAL_SEED_COUNT} ({eval_seed}) ... ",
                end="",
                flush=True,
            )
            result = score_ckpt(
                ckpt_path=str(ckpt_path),
                seed=eval_seed,
                max_steps=eval_max_steps,
                episodes=episodes_per_seed,
                workers=workers,
                capture_hz=capture_hz,
            )
            print(
                f"score={result['mean_score']:.4f} "
                f"placement={result['placement_success_rate']:.4f}"
            )
            seed_results.append({
                "seed": eval_seed,
                "mean_score": result["mean_score"],
                "grasp_rate": result["grasp_rate"],
                "lift_rate": result["lift_rate"],
                "tray_reach_rate": result["tray_reach_rate"],
                "lowered_to_tray_rate": result["lowered_to_tray_rate"],
                "released_over_tray_rate": result["released_over_tray_rate"],
                "placement_success_rate": result["placement_success_rate"],
                "eval_metric_version": result["eval_metric_version"],
            })

        final_result = aggregate_seed_results(seed_results=seed_results)
        print(
            f"    Aggregate: score={float(final_result['mean_score']):.4f} "
            f"placement={float(final_result['placement_success_rate']):.4f}"
        )

        final_rounds.append({
            "round": round_idx,
            "checkpoint": str(ckpt_path),
            "original_score": original_score,
            "original_placement_success_rate": original_placement,
            "final_score": final_result["mean_score"],
            "final_placement_success_rate": final_result["placement_success_rate"],
            "final_metrics": {
                "grasp_rate": final_result["grasp_rate"],
                "lift_rate": final_result["lift_rate"],
                "tray_reach_rate": final_result["tray_reach_rate"],
                "lowered_to_tray_rate": final_result["lowered_to_tray_rate"],
                "released_over_tray_rate": final_result["released_over_tray_rate"],
                "placement_success_rate": final_result["placement_success_rate"],
            },
            "per_seed_metrics": seed_results,
            "eval_metric_version": final_result["eval_metric_version"],
        })

    final_rounds.sort(key=lambda x: x["round"])

    best_by_score = max(
        final_rounds,
        key=lambda item: eval_selection_key(
            selection_mode=selection_mode,
            mean_score=float(item["final_score"]),
            placement_success_rate=float(item["final_placement_success_rate"]),
        ),
    )

    print(f"\n{'=' * SECTION_WIDTH}")
    print(f"{'Round':<8}{'Orig Score':<12}{'Final Score':<14}{'Orig Place':<12}{'Final Place':<14}Checkpoint")
    print(f"{'-' * SECTION_WIDTH}")
    for r in final_rounds:
        orig_s = f'{r["original_score"]:.4f}' if r["original_score"] is not None else "-"
        orig_p = f'{r["original_placement_success_rate"]:.4f}' if r["original_placement_success_rate"] is not None else "-"
        print(f'{r["round"]:<8d}{orig_s:<12}{r["final_score"]:<14.4f}{orig_p:<12}{r["final_placement_success_rate"]:<14.4f}{r["checkpoint"]}')

    print(f"\nBest by final {selection_mode}: round {best_by_score['round']:03d}, "
          f"score {best_by_score['final_score']:.4f}, "
          f"placement {best_by_score['final_placement_success_rate']:.4f}")
    print(f"Checkpoint: {best_by_score['checkpoint']}")

    output = {
        "run_name": run_name,
        "original_eval_seed": metrics_data.get("eval_seed", 42),
        "final_eval_seed": final_eval_seed,
        "final_eval_seeds": final_eval_seeds,
        "seed_count": FINAL_EVAL_SEED_COUNT,
        "eval_episodes": eval_episodes,
        "episodes_per_seed": episodes_per_seed,
        "eval_max_steps": eval_max_steps,
        "workers": workers,
        "capture_hz": capture_hz,
        "eval_selection_mode": selection_mode,
        "overall_best_round": best_by_score["round"],
        "overall_best_checkpoint": best_by_score["checkpoint"],
        "overall_best_score": best_by_score["final_score"],
        "overall_best_placement_success_rate": best_by_score["final_placement_success_rate"],
        "rounds": final_rounds,
    }

    output_path = run_results_dir / "final_scores.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nSaved: {output_path}")

    plot_comparison(
        final_rounds=final_rounds,
        run_name=run_name,
        save_dir=run_results_dir,
        eval_episodes=eval_episodes,
        original_episodes=original_episodes,
    )


def plot_comparison(
    *,
    final_rounds: list[dict],
    run_name: str,
    save_dir: Path,
    eval_episodes: int,
    original_episodes: int,
) -> None:
    rounds_arr = np.array([r["round"] for r in final_rounds])
    orig_scores = np.array([
        r["original_score"] if r["original_score"] is not None else np.nan
        for r in final_rounds
    ])
    final_scores = np.array([r["final_score"] for r in final_rounds])
    orig_place = np.array([
        r["original_placement_success_rate"] if r["original_placement_success_rate"] is not None else np.nan
        for r in final_rounds
    ])
    final_place = np.array([r["final_placement_success_rate"] for r in final_rounds])

    score_fig, score_ax = plt.subplots(figsize=(10, 5))
    score_ax.plot(
        rounds_arr,
        orig_scores,
        "o-",
        color="#888888",
        label=f"Flywheel evaluation ({original_episodes} ep)",
        linewidth=1.5,
        markersize=6,
    )
    score_ax.plot(
        rounds_arr,
        final_scores,
        "s-",
        color="#1f77b4",
        label=f"Large held-out evaluation ({eval_episodes} ep)",
        linewidth=1.5,
        markersize=6,
    )
    score_ax.set_xlabel("Round")
    score_ax.set_ylabel("Mean score")
    score_ax.set_title(f"Flywheel versus held-out evaluation - {run_name}")
    score_ax.set_xticks(rounds_arr)
    score_ax.set_xticklabels([f"{round_index:03d}" for round_index in rounds_arr])
    score_ax.set_ylim(0, 1.05)
    score_ax.grid(True, alpha=0.3)
    score_ax.legend()
    score_fig.tight_layout()
    score_curve_path = save_dir / "final-score-curve.png"
    score_fig.savefig(score_curve_path, dpi=160)
    plt.close(score_fig)
    print(f"Plot saved: {score_curve_path}")

    place_fig, place_ax = plt.subplots(figsize=(10, 5))
    place_ax.plot(
        rounds_arr,
        orig_place,
        "o-",
        color="#888888",
        label=f"Flywheel evaluation ({original_episodes} ep)",
        linewidth=1.5,
        markersize=6,
    )
    place_ax.plot(
        rounds_arr,
        final_place,
        "s-",
        color="#2ca02c",
        label=f"Large held-out evaluation ({eval_episodes} ep)",
        linewidth=1.5,
        markersize=6,
    )
    place_ax.set_xlabel("Round")
    place_ax.set_ylabel("Placement success rate")
    place_ax.set_title(f"Flywheel versus held-out evaluation - placement - {run_name}")
    place_ax.set_xticks(rounds_arr)
    place_ax.set_xticklabels([f"{round_index:03d}" for round_index in rounds_arr])
    place_ax.set_ylim(0, 1.05)
    place_ax.grid(True, alpha=0.3)
    place_ax.legend()
    place_fig.tight_layout()
    place_curve_path = save_dir / "final-placement-score.png"
    place_fig.savefig(place_curve_path, dpi=160)
    plt.close(place_fig)
    print(f"Plot saved: {place_curve_path}")


if __name__ == "__main__":
    main()
