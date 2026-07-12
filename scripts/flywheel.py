""" expert + dagger data flywheel """

import argparse
import json
import re
import secrets
from pathlib import Path

import torch

from train import train
from rollout import rollout
from tqdm import tqdm

def next_flywheel_run_name(*, root: Path) -> str:
    pattern = re.compile(r"^run-(\d+)$")
    indices = []
    if root.exists():
        indices = [
            int(match.group(1))
            for path in root.iterdir()
            if path.is_dir() and (match := pattern.match(path.name))
        ]

    return f"run-{max(indices, default=0) + 1:03d}"


def make_beta_schedule(
    *,
    beta_start: float,
    beta_final: float,
    num_dagger_rounds: int,) -> list[float]:
    if num_dagger_rounds == 1:
        return [beta_start]

    return [
        beta_start
        + round_index * (beta_final - beta_start) / (num_dagger_rounds - 1)
        for round_index in range(num_dagger_rounds)
    ]


def append_round_metrics(
    *,
    metrics_path: Path,
    run_name: str,
    rounds: list[dict[str, object]],
    round_index: int,
    beta: float | None,
    dagger_seed: int | None,
    data_dirs: list[Path],
    sample_ratios: list[float],
    best_checkpoint: Path,
    config: dict[str, object],
    eval_seed: int,
    eval_episodes: int,
    eval_max_steps: int,) -> None:
    checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    rounds.append({
        "round": round_index,
        "beta": beta,
        "dagger_seed": dagger_seed,
        "data_dirs": [str(path) for path in data_dirs],
        "sample_ratios": sample_ratios,
        "best_checkpoint": str(best_checkpoint),
        "best_epoch": int(checkpoint["epoch"]),
        "best_score": float(checkpoint["eval_score"]),
    })

    overall_best = max(rounds, key=lambda item: float(item["best_score"]))
    metrics = {
        "run_name": run_name,
        "eval_seed": eval_seed,
        "eval_episodes": eval_episodes,
        "eval_max_steps": eval_max_steps,
        "config": config,
        "rounds": rounds,
        "overall_best_checkpoint": overall_best["best_checkpoint"],
        "overall_best_score": overall_best["best_score"],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


def run_flywheel(
    *,
    run_name: str,
    num_dagger_rounds: int,
    beta_start: float,
    beta_final: float,
    expert_npz_dir: Path,
    num_epochs: int,
    dagger_episodes: int,
    rollout_max_steps: int,
    eval_interval: int,
    eval_seed: int,
    eval_episodes: int,
    eval_max_steps: int,
    early_stop_patience: int,
    expert_ratio: float,
    train_seed: int,) -> None:

    ckpt_root = Path('checkpoints/flywheel') / run_name
    runs_root = Path('runs/flywheel') / run_name
    results_root = Path('results/flywheel') / run_name
    results_root.mkdir(parents=True, exist_ok=True)
    metrics_path = results_root / "metrics.json"

    beta_schedule = make_beta_schedule(
        beta_start=beta_start,
        beta_final=beta_final,
        num_dagger_rounds=num_dagger_rounds,
    )
    dagger_dirs: list[Path] = []
    round_metrics: list[dict[str, object]] = []
    config = {
        "expert_dir": str(expert_npz_dir),
        "epochs": num_epochs,
        "dagger_rounds": num_dagger_rounds,
        "beta_start": beta_start,
        "beta_final": beta_final,
        "beta_schedule": beta_schedule,
        "dagger_episodes": dagger_episodes,
        "rollout_max_steps": rollout_max_steps,
        "eval_interval": eval_interval,
        "eval_seed": eval_seed,
        "eval_episodes": eval_episodes,
        "eval_max_steps": eval_max_steps,
        "early_stop_patience": early_stop_patience,
        "expert_ratio": expert_ratio,
        "train_seed": train_seed,
    }

    num_rounds = num_dagger_rounds + 1
    for round in tqdm(range(num_rounds), desc="Running flywheel"):
        print(f"Running round #{round}/{num_rounds - 1}")

        if round == 0:
            # train expert-only policy
            ckpt_dir = ckpt_root / f"round-{round}"
            runs_dir = runs_root / f"round-{round}"

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            runs_dir.mkdir(parents=True, exist_ok=True)

            torch.manual_seed(train_seed)
            train(num_epochs=num_epochs,
                npz_folders=[expert_npz_dir],
                checkpoint_dir=ckpt_dir,
                log_dir=runs_dir,
                sample_seed=train_seed,
                eval_interval=eval_interval,
                eval_seed=eval_seed,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
                early_stop_patience=early_stop_patience,
                )
            append_round_metrics(
                metrics_path=metrics_path,
                run_name=run_name,
                rounds=round_metrics,
                round_index=round,
                beta=None,
                dagger_seed=None,
                data_dirs=[expert_npz_dir],
                sample_ratios=[1.0],
                best_checkpoint=ckpt_dir / "best.pt",
                config=config,
                eval_seed=eval_seed,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
            )

        else:
            beta = beta_schedule[round - 1]
            dagger_seed = secrets.randbelow(2**32)
            print(f"DAgger seed: {dagger_seed}")

            # generate dagger data
            dagger_dir = Path('data/dagger/flywheel') / run_name / f"round-{round}"
            model_path = ckpt_root / f"round-{round-1}" / "best.pt"
            print(f"model_path: {model_path}")

            # use dagger to collect data
            rollout(model_path=model_path,
                    randomize_scene=True,
                    seed=dagger_seed,
                    episodes=dagger_episodes,
                    max_steps=rollout_max_steps,
                    dagger=True,
                    dagger_root=dagger_dir,
                    beta=beta,
                    create_dagger_subdir=False,
                    headless=True,
                    log_root=None,
                    train_npz_dir=None,
                    log_rollout=False,
                    )

            # retrain with dagger data
            dagger_dirs.append(dagger_dir)
            ckpt_dir = ckpt_root / f"round-{round}"
            runs_dir = runs_root / f"round-{round}"

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            runs_dir.mkdir(parents=True, exist_ok=True)

            dagger_ratio = (1.0 - expert_ratio) / len(dagger_dirs)
            sample_ratios = [
                expert_ratio,
                *[dagger_ratio] * len(dagger_dirs),
            ]

            torch.manual_seed(train_seed)
            train(num_epochs=num_epochs,
                npz_folders=[expert_npz_dir, *dagger_dirs],
                checkpoint_dir=ckpt_dir,
                log_dir=runs_dir,
                sample_ratios=sample_ratios,
                sample_seed=train_seed,
                eval_interval=eval_interval,
                eval_seed=eval_seed,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
                early_stop_patience=early_stop_patience,
                )
            append_round_metrics(
                metrics_path=metrics_path,
                run_name=run_name,
                rounds=round_metrics,
                round_index=round,
                beta=beta,
                dagger_seed=dagger_seed,
                data_dirs=[expert_npz_dir, *dagger_dirs],
                sample_ratios=sample_ratios,
                best_checkpoint=ckpt_dir / "best.pt",
                config=config,
                eval_seed=eval_seed,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
            )



def parse_args():
    parser = argparse.ArgumentParser(description="Run the expert and DAgger data flywheel")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--expert-dir",
        type=Path,
        default=Path("data/expert/rand-100"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--beta-start", type=float, default=1.0)
    parser.add_argument("--beta-final", type=float, default=0.0)
    parser.add_argument("--dagger-episodes", type=int, default=10)
    parser.add_argument("--rollout-max-steps", type=int, default=1400)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-max-steps", type=int, default=1400)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--expert-ratio", type=float, default=0.5)
    parser.add_argument("--train-seed", type=int, default=0)
    args = parser.parse_args()

    if not args.expert_dir.is_dir():
        parser.error(f"--expert-dir does not exist or is not a directory: {args.expert_dir}")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.dagger_rounds < 1:
        parser.error("--dagger-rounds must be at least 1")
    if not 0.0 <= args.beta_start <= 1.0:
        parser.error("--beta-start must be in [0, 1]")
    if not 0.0 <= args.beta_final <= 1.0:
        parser.error("--beta-final must be in [0, 1]")
    if args.beta_start < args.beta_final:
        parser.error("--beta-start must be greater than or equal to --beta-final")
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
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be non-negative")
    if not 0.0 < args.expert_ratio < 1.0:
        parser.error("--expert-ratio must be between 0 and 1")

    return args


def main():
    args = parse_args()
    run_name = args.run_name or next_flywheel_run_name(
        root=Path("data/dagger/flywheel")
    )
    print(f"Running flywheel in {run_name}")
    run_flywheel(
        run_name=run_name,
        num_dagger_rounds=args.dagger_rounds,
        beta_start=args.beta_start,
        beta_final=args.beta_final,
        expert_npz_dir=args.expert_dir,
        num_epochs=args.epochs,
        dagger_episodes=args.dagger_episodes,
        rollout_max_steps=args.rollout_max_steps,
        eval_interval=args.eval_interval,
        eval_seed=args.eval_seed,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        early_stop_patience=args.early_stop_patience,
        expert_ratio=args.expert_ratio,
        train_seed=args.train_seed,
    )
if __name__ == "__main__":
    main()
