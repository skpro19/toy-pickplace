""" expert + dagger data flywheel """

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from dagger_metrics import (
    plot_dagger_round,
    plot_dagger_round_trends,
    summarize_dagger_round,
    write_dagger_metrics,
)
from eval import (
    DEFAULT_EVAL_SELECTION_MODE,
    EVAL_SELECTION_MODES,
    EvalSelectionMode,
    eval_selection_key,
)
from train import train
from rollout import rollout
from data import collect_expert_episodes


SECTION_WIDTH = 72


def format_duration(*, seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def print_section(*, title: str) -> None:
    print()
    print("=" * SECTION_WIDTH)
    print(title)
    print("=" * SECTION_WIDTH)


def print_final_summary(
    *,
    rounds: list[dict[str, object]],
    metrics_path: Path,
    selection_mode: EvalSelectionMode,
) -> None:
    print_section(title="Final summary")
    print(
        f"{'Round':<8}{'Threshold':<12}{'Epoch':<10}{'Placement':<12}"
        f"{'Score':<10}Checkpoint"
    )
    print("-" * SECTION_WIDTH)
    for item in rounds:
        round_label = f'{int(item["round"]):03d}'
        threshold = (
            "-"
            if item["intervention_threshold"] is None
            else f'{float(item["intervention_threshold"]):.3f}'
        )
        print(
            f'{round_label:<8}{threshold:<12}{int(item["best_epoch"]):<10}'
            f'{float(item["best_placement_success_rate"]):<12.4f}'
            f'{float(item["best_score"]):<10.4f}{item["best_checkpoint"]}'
        )

    overall_best = select_best_round(rounds=rounds, selection_mode=selection_mode)
    print()
    print(
        f'Overall best: round {int(overall_best["round"]):03d}, '
        f'placement {float(overall_best["best_placement_success_rate"]):.4f}, '
        f'score {float(overall_best["best_score"]):.4f}, mode {selection_mode}'
    )
    print(f'Checkpoint: {overall_best["best_checkpoint"]}')
    print(f"Metrics: {metrics_path}")

def next_flywheel_run_name(*, root: Path, occupied_roots: list[Path]) -> str:
    pattern = re.compile(r"^run-(\d+)$")
    root.mkdir(parents=True, exist_ok=True)

    while True:
        indices = [
            int(match.group(1))
            for occupied_root in [root, *occupied_roots]
            if occupied_root.exists()
            for path in occupied_root.iterdir()
            if (match := pattern.match(path.name))
        ]
        run_name = f"run-{max(indices, default=0) + 1:03d}"

        try:
            # mkdir without exist_ok reserves this name across concurrent processes.
            (root / run_name).mkdir()
        except FileExistsError:
            continue

        return run_name


def make_dagger_round_seeds(*, seed: int, rounds: int) -> list[int]:
    seed_sequence = np.random.SeedSequence(seed)
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(rounds)
    ]


def make_flywheel_seeds(*, global_seed: int) -> dict[str, int]:
    names = (
        "expert_seed",
        "train_seed",
        "dagger_seed",
        "eval_seed",
    )
    seed_sequence = np.random.SeedSequence(global_seed)
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(names, seed_sequence.spawn(len(names)), strict=True)
    }


def round_selection_key(
    *,
    item: dict[str, object],
    selection_mode: EvalSelectionMode,
) -> tuple[float, ...]:
    return eval_selection_key(
        selection_mode=selection_mode,
        mean_score=float(item["best_score"]),
        placement_success_rate=float(item["best_placement_success_rate"]),
    )


def select_best_round(
    *,
    rounds: list[dict[str, object]],
    selection_mode: EvalSelectionMode,
) -> dict[str, object]:
    return max(
        rounds,
        key=lambda item: round_selection_key(
            item=item,
            selection_mode=selection_mode,
        ),
    )


def append_round_metrics(
    *,
    metrics_path: Path,
    run_name: str,
    rounds: list[dict[str, object]],
    round_index: int,
    intervention_threshold: float | None,
    dagger_seed: int | None,
    dagger_metrics: dict[str, object] | None,
    data_dirs: list[Path],
    sample_ratios: list[float],
    best_checkpoint: Path,
    config: dict[str, object],
    eval_seed: int,
    eval_episodes: int,
    eval_max_steps: int,
    selection_mode: EvalSelectionMode,
) -> dict[str, object]:
    checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    eval_metric_version = int(checkpoint.get("eval_metric_version", 1))
    checkpoint_selection_mode = checkpoint.get(
        "eval_selection_mode",
        DEFAULT_EVAL_SELECTION_MODE,
    )
    if checkpoint_selection_mode != selection_mode:
        raise ValueError(
            "Checkpoint evaluation selection mode does not match flywheel mode"
        )
    existing_versions = {
        int(item.get("eval_metric_version", 1)) for item in rounds
    }
    if existing_versions and existing_versions != {eval_metric_version}:
        raise ValueError("Cannot compare flywheel scores from different metric versions")

    eval_metrics = checkpoint.get("eval_metrics", {})
    if "placement_success_rate" not in eval_metrics:
        raise ValueError("Checkpoint is missing placement_success_rate")

    round_metrics = {
        "round": round_index,
        "intervention_threshold": intervention_threshold,
        "dagger_seed": dagger_seed,
        "dagger_metrics": dagger_metrics,
        "data_dirs": [str(path) for path in data_dirs],
        "sample_ratios": sample_ratios,
        "best_checkpoint": str(best_checkpoint),
        "best_epoch": int(checkpoint["epoch"]),
        "best_score": float(checkpoint["eval_score"]),
        "best_placement_success_rate": float(
            eval_metrics["placement_success_rate"]
        ),
        "eval_metric_version": eval_metric_version,
        "eval_selection_mode": selection_mode,
        "eval_metrics": eval_metrics,
    }
    rounds.append(round_metrics)

    overall_best = select_best_round(
        rounds=rounds,
        selection_mode=selection_mode,
    )
    metrics = {
        "run_name": run_name,
        "global_seed": config.get("global_seed"),
        "eval_seed": eval_seed,
        "eval_episodes": eval_episodes,
        "eval_max_steps": eval_max_steps,
        "eval_metric_version": eval_metric_version,
        "eval_selection_mode": selection_mode,
        "config": config,
        "rounds": rounds,
        "overall_best_checkpoint": overall_best["best_checkpoint"],
        "overall_best_score": overall_best["best_score"],
        "overall_best_placement_success_rate": overall_best[
            "best_placement_success_rate"
        ],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return round_metrics


def expert_npz_dir_for_run(*, run_name: str) -> Path:
    return Path("data/flywheel") / run_name / "expert"


def expert_save_images_for_arch(*, arch: str) -> bool:
    return arch == "vision_mlp"


def validate_expert_npz_for_arch(
    *,
    expert_npz_dir: Path,
    arch: str,
) -> None:
    if not expert_save_images_for_arch(arch=arch):
        return

    npz_files = sorted(expert_npz_dir.glob("*.npz"))
    if not npz_files:
        return

    missing_img_obs = []
    for path in npz_files:
        with np.load(path) as data:
            if "img_obs" not in data:
                missing_img_obs.append(path.name)
    if not missing_img_obs:
        return

    raise ValueError(
        f"Expert data in {expert_npz_dir} lacks img_obs required for "
        f"arch={arch!r} ({len(missing_img_obs)} of {len(npz_files)} files). "
        "Delete the expert directory or use a new --run-name."
    )


def run_flywheel(
    *,
    run_name: str,
    num_dagger_rounds: int,
    intervention_threshold: float,
    intervention_steps: int,
    num_expert_episodes: int,
    max_steps: int,
    num_epochs: int,
    batch_size: int,
    dagger_episodes: int,
    rollout_max_steps: int,
    eval_interval: int,
    eval_seed: int,
    eval_episodes: int,
    eval_max_steps: int,
    workers: int,
    dataloader_workers: int = 0,
    early_stop_patience: int,
    expert_ratio: float,
    dagger_intervention_ratio: float,
    global_seed: int,
    expert_seed: int,
    train_seed: int,
    dagger_seed: int,
    arch: str,
    eval_selection_mode: EvalSelectionMode = DEFAULT_EVAL_SELECTION_MODE,
) -> None:

    ckpt_root = Path('checkpoints/flywheel') / run_name
    runs_root = Path('runs/flywheel') / run_name
    results_root = Path('results/flywheel') / run_name
    results_root.mkdir(parents=True, exist_ok=True)
    metrics_path = results_root / "metrics.json"
    expert_npz_dir = expert_npz_dir_for_run(run_name=run_name)

    save_expert_images = expert_save_images_for_arch(arch=arch)

    print_section(title=f"Flywheel {run_name}: expert collection")
    print(
        f"Episodes: {num_expert_episodes} | Max steps: {max_steps} | "
        f"Seed: {expert_seed}"
    )
    print(
        f"Policy arch: {arch} | Expert images: "
        f"{'yes' if save_expert_images else 'no'}"
    )
    print(f"Output: {expert_npz_dir}")
    collection_started_at = time.perf_counter()
    collect_expert_episodes(
        episodes=num_expert_episodes,
        out_dir=expert_npz_dir,
        seed=expert_seed,
        max_steps=max_steps,
        save_images=save_expert_images,
    )
    validate_expert_npz_for_arch(
        expert_npz_dir=expert_npz_dir,
        arch=arch,
    )
    print(
        f"Expert collection complete | elapsed: "
        f"{format_duration(seconds=time.perf_counter() - collection_started_at)}"
    )

    dagger_round_seeds = make_dagger_round_seeds(
        seed=dagger_seed,
        rounds=num_dagger_rounds,
    )
    dagger_dirs: list[Path] = []
    dagger_summaries: list[dict[str, object]] = []
    round_metrics: list[dict[str, object]] = []
    config = {
        "num_expert_episodes": num_expert_episodes,
        "max_steps": max_steps,
        "expert_dir": str(expert_npz_dir),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "dagger_rounds": num_dagger_rounds,
        "dagger_mode": "threshold",
        "intervention_threshold": intervention_threshold,
        "intervention_steps": intervention_steps,
        "dagger_episodes": dagger_episodes,
        "rollout_max_steps": rollout_max_steps,
        "eval_interval": eval_interval,
        "eval_seed": eval_seed,
        "eval_episodes": eval_episodes,
        "eval_max_steps": eval_max_steps,
        "eval_selection_mode": eval_selection_mode,
        "workers": workers,
        "dataloader_workers": dataloader_workers,
        "early_stop_patience": early_stop_patience,
        "expert_ratio": expert_ratio,
        "dagger_intervention_ratio": dagger_intervention_ratio,
        "global_seed": global_seed,
        "expert_seed": expert_seed,
        "train_seed": train_seed,
        "dagger_seed": dagger_seed,
        "arch": arch,
    }

    print_section(title=f"Flywheel {run_name}")
    print(f"Policy arch: {arch}")
    print(
        f"Expert data: {expert_npz_dir} "
        f"({num_expert_episodes} episodes, max_steps={max_steps})"
    )
    print(
        f"DAgger rounds: {num_dagger_rounds} | Max epochs: {num_epochs} | "
        f"Batch size: {batch_size} | "
        f"DataLoader workers: {dataloader_workers}"
    )
    print(
        f"Expert ratio: {expert_ratio:.2f} | "
        f"DAgger intervention ratio: {dagger_intervention_ratio:.2f} | "
        f"Threshold: {intervention_threshold:.3f} | "
        f"Intervention steps: {intervention_steps}"
    )
    print(
        f"Evaluation: {eval_episodes} episodes x {eval_max_steps} steps "
        f"every {eval_interval} epochs | selection: {eval_selection_mode}"
    )
    print(f"Metrics: {metrics_path}")

    num_rounds = num_dagger_rounds + 1
    flywheel_started_at = time.perf_counter()
    for round in range(num_rounds):
        round_name = f"round-{round:03d}"
        round_started_at = time.perf_counter()

        if round == 0:
            # train expert-only policy
            ckpt_dir = ckpt_root / round_name
            runs_dir = runs_root / round_name

            print_section(
                title=f"Round {round:03d}/{num_rounds - 1:03d}: expert training"
            )
            print(f"Dataset: {expert_npz_dir}")
            print(f"Checkpoint: {ckpt_dir}")
            print(f"TensorBoard: {runs_dir}")

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            runs_dir.mkdir(parents=True, exist_ok=True)

            torch.manual_seed(train_seed)
            train(
                arch=arch,
                num_epochs=num_epochs,
                batch_size=batch_size,
                npz_folders=[expert_npz_dir],
                checkpoint_dir=ckpt_dir,
                log_dir=runs_dir,
                sample_seed=train_seed,
                eval_interval=eval_interval,
                eval_seed=eval_seed,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
                eval_workers=workers,
                dataloader_workers=dataloader_workers,
                early_stop_patience=early_stop_patience,
                eval_selection_mode=eval_selection_mode,
            )
            metrics = append_round_metrics(
                metrics_path=metrics_path,
                run_name=run_name,
                rounds=round_metrics,
                round_index=round,
                intervention_threshold=None,
                dagger_seed=None,
                dagger_metrics=None,
                data_dirs=[expert_npz_dir],
                sample_ratios=[1.0],
                best_checkpoint=ckpt_dir / "best.pt",
                config=config,
                eval_seed=eval_seed,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
                selection_mode=eval_selection_mode,
            )
            print(
                f'Round complete | best epoch: {int(metrics["best_epoch"])} | '
                f'placement: {float(metrics["best_placement_success_rate"]):.4f} | '
                f'score: {float(metrics["best_score"]):.4f} | '
                f'elapsed: {format_duration(seconds=time.perf_counter() - round_started_at)}'
            )

        else:
            round_dagger_seed = dagger_round_seeds[round - 1]

            # generate dagger data
            dagger_dir = Path("data/flywheel") / run_name / round_name / "dagger"
            previous_round_name = f"round-{round - 1:03d}"
            model_path = ckpt_root / previous_round_name / "best.pt"

            print_section(
                title=f"Round {round:03d}/{num_rounds - 1:03d}: DAgger collection"
            )
            print(f"Policy: {model_path}")
            print(
                f"Threshold: {intervention_threshold:.3f} | "
                f"Intervention steps: {intervention_steps} | "
                f"Seed: {round_dagger_seed}"
            )
            print(
                f"Episodes: {dagger_episodes} | "
                f"Max steps: {rollout_max_steps}"
            )
            print(f"Output: {dagger_dir}")

            # use dagger to collect data
            collection_started_at = time.perf_counter()
            rollout(
                model_path=model_path,
                randomize_scene=True,
                seed=round_dagger_seed,
                episodes=dagger_episodes,
                max_steps=rollout_max_steps,
                dagger=True,
                dagger_root=dagger_dir,
                beta=0.0,
                intervention_mode="threshold",
                intervention_threshold=intervention_threshold,
                intervention_steps=intervention_steps,
                create_dagger_subdir=False,
                headless=True,
                log_root=None,
                train_npz_dir=None,
                log_rollout=False,
                workers=workers,
            )
            dagger_summary, dagger_episodes_data = summarize_dagger_round(
                dagger_dir=dagger_dir,
                round_index=round,
            )
            write_dagger_metrics(
                metrics_path=dagger_dir / "metrics.json",
                metrics=dagger_summary,
            )
            plots_dir = results_root / "plots"
            plot_dagger_round(
                episodes=dagger_episodes_data,
                summary=dagger_summary,
                save_path=plots_dir / f"{round_name}-dagger.png",
            )
            dagger_summaries.append(dagger_summary)
            plot_dagger_round_trends(
                round_summaries=dagger_summaries,
                save_path=plots_dir / "dagger-round-trends.png",
            )
            print(
                f"Collection complete | elapsed: "
                f"{format_duration(seconds=time.perf_counter() - collection_started_at)}"
            )
            print(
                f'Expert actions: {float(dagger_summary["expert_action_fraction"]):.1%} | '
                f'Triggers: '
                f'{float(dagger_summary["intervention_trigger_fraction"]):.1%} | '
                f'Segments: {int(dagger_summary["expert_control_segments"])}'
            )

            # retrain with dagger data
            dagger_dirs.append(dagger_dir)
            ckpt_dir = ckpt_root / round_name
            runs_dir = runs_root / round_name

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            runs_dir.mkdir(parents=True, exist_ok=True)

            dagger_ratio = (1.0 - expert_ratio) / len(dagger_dirs)
            sample_ratios = [
                expert_ratio,
                *[dagger_ratio] * len(dagger_dirs),
            ]

            print_section(
                title=f"Round {round:03d}/{num_rounds - 1:03d}: retraining"
            )
            print(f"Datasets: {len(dagger_dirs) + 1}")
            print(f"Checkpoint: {ckpt_dir}")
            print(f"TensorBoard: {runs_dir}")

            torch.manual_seed(train_seed)
            train(
                arch=arch,
                num_epochs=num_epochs,
                batch_size=batch_size,
                npz_folders=[expert_npz_dir, *dagger_dirs],
                checkpoint_dir=ckpt_dir,
                log_dir=runs_dir,
                sample_ratios=sample_ratios,
                dagger_intervention_ratio=dagger_intervention_ratio,
                sample_seed=train_seed,
                eval_interval=eval_interval,
                eval_seed=eval_seed,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
                eval_workers=workers,
                dataloader_workers=dataloader_workers,
                early_stop_patience=early_stop_patience,
                eval_selection_mode=eval_selection_mode,
            )
            metrics = append_round_metrics(
                metrics_path=metrics_path,
                run_name=run_name,
                rounds=round_metrics,
                round_index=round,
                intervention_threshold=intervention_threshold,
                dagger_seed=round_dagger_seed,
                dagger_metrics=dagger_summary,
                data_dirs=[expert_npz_dir, *dagger_dirs],
                sample_ratios=sample_ratios,
                best_checkpoint=ckpt_dir / "best.pt",
                config=config,
                eval_seed=eval_seed,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
                selection_mode=eval_selection_mode,
            )
            print(
                f'Round complete | best epoch: {int(metrics["best_epoch"])} | '
                f'placement: {float(metrics["best_placement_success_rate"]):.4f} | '
                f'score: {float(metrics["best_score"]):.4f} | '
                f'elapsed: {format_duration(seconds=time.perf_counter() - round_started_at)}'
            )

    print_final_summary(
        rounds=round_metrics,
        metrics_path=metrics_path,
        selection_mode=eval_selection_mode,
    )
    print(
        f"Total elapsed: "
        f"{format_duration(seconds=time.perf_counter() - flywheel_started_at)}"
    )



def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, required=True)
    config_args, _ = config_parser.parse_known_args()

    config: dict[str, object] = {}
    if config_args.config is not None:
        try:
            loaded_config = yaml.safe_load(config_args.config.read_text())
        except (OSError, yaml.YAMLError) as error:
            config_parser.error(f"could not load --config: {error}")
        if loaded_config is not None and not isinstance(loaded_config, dict):
            config_parser.error("--config must contain a YAML mapping")
        config = loaded_config or {}
        if any(not isinstance(key, str) for key in config):
            config_parser.error("--config keys must be strings")

    parser = argparse.ArgumentParser(description="Run the expert and DAgger data flywheel")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-expert-episodes", type=int, default=100)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8000,
        help="Maximum steps per scripted expert demonstration episode",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dagger-rounds", type=int, default=10)
    parser.add_argument(
        "--intervention-threshold",
        type=float,
        default=0.2,
        help="L2 arm-action threshold; gripper disagreement also triggers control",
    )
    parser.add_argument(
        "--intervention-steps",
        type=int,
        default=50,
        help="Minimum expert-control burst after an arm or gripper trigger",
    )
    parser.add_argument("--dagger-episodes", type=int, default=50)
    parser.add_argument("--rollout-max-steps", type=int, default=1400)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--eval-max-steps", type=int, default=1400)
    parser.add_argument(
        "--mode",
        choices=EVAL_SELECTION_MODES,
        default=DEFAULT_EVAL_SELECTION_MODE,
        help="Select checkpoints by weighted score or placement rate first",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--expert-ratio", type=float, default=0.5)
    parser.add_argument(
        "--dagger-intervention-ratio",
        type=float,
        default=0.8,
        help="Sampling share for execute_expert frames within DAgger data",
    )
    parser.add_argument(
        "--global-seed",
        type=int,
        default=0,
        help="Root seed used to derive expert, training, DAgger, and evaluation seeds",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["mlp", "vision_mlp"],
        help="Policy architecture used for flywheel training rounds",
    )

    valid_config_keys = {
        action.dest for action in parser._actions if action.dest not in {"help", "config"}
    }
    unknown_config_keys = set(config) - valid_config_keys
    if unknown_config_keys:
        parser.error(
            f"unknown --config keys: {sorted(unknown_config_keys)}"
        )
    parser.set_defaults(**config)
    args = parser.parse_args()

    if args.arch is None:
        parser.error("arch is required (set arch in --config or pass --arch)")

    if args.num_expert_episodes < 1:
        parser.error("--num-expert-episodes must be at least 1")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.dagger_rounds < 1:
        parser.error("--dagger-rounds must be at least 1")
    if args.intervention_threshold < 0.0:
        parser.error("--intervention-threshold must be non-negative")
    if args.intervention_steps < 1:
        parser.error("--intervention-steps must be at least 1")
    if args.dagger_episodes < 1:
        parser.error("--dagger-episodes must be at least 1")
    if args.rollout_max_steps < 1:
        parser.error("--rollout-max-steps must be at least 1")
    if args.eval_interval < 1:
        parser.error("--eval-interval must be at least 1")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be at least 1")
    if args.eval_max_steps < 1:
        parser.error("--eval-max-steps must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.dataloader_workers < 0:
        parser.error("--dataloader-workers must be non-negative")
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be non-negative")
    if not 0.0 < args.expert_ratio < 1.0:
        parser.error("--expert-ratio must be between 0 and 1")
    if not 0.0 <= args.dagger_intervention_ratio <= 1.0:
        parser.error("--dagger-intervention-ratio must be between 0 and 1")
    if args.global_seed < 0:
        parser.error("--global-seed must be non-negative")
    if args.mode not in EVAL_SELECTION_MODES:
        parser.error(f"--mode must be one of {EVAL_SELECTION_MODES}")

    return args


def main():
    args = parse_args()
    seeds = make_flywheel_seeds(global_seed=args.global_seed)
    run_name = args.run_name or next_flywheel_run_name(
        root=Path("data/flywheel"),
        occupied_roots=[
            Path("checkpoints/flywheel"),
            Path("runs/flywheel"),
            Path("results/flywheel"),
        ],
    )
    run_flywheel(
        run_name=run_name,
        num_dagger_rounds=args.dagger_rounds,
        intervention_threshold=args.intervention_threshold,
        intervention_steps=args.intervention_steps,
        num_expert_episodes=args.num_expert_episodes,
        max_steps=args.max_steps,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        dagger_episodes=args.dagger_episodes,
        rollout_max_steps=args.rollout_max_steps,
        eval_interval=args.eval_interval,
        eval_seed=seeds["eval_seed"],
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        workers=args.workers,
        dataloader_workers=args.dataloader_workers,
        early_stop_patience=args.early_stop_patience,
        expert_ratio=args.expert_ratio,
        dagger_intervention_ratio=args.dagger_intervention_ratio,
        global_seed=args.global_seed,
        expert_seed=seeds["expert_seed"],
        train_seed=seeds["train_seed"],
        dagger_seed=seeds["dagger_seed"],
        arch=args.arch,
        eval_selection_mode=args.mode,
    )
if __name__ == "__main__":
    main()
