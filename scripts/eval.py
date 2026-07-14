"""Score model checkpoints using physical task milestones."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch

from rollout import (
    TaskMetrics,
    load_policy,
    make_episode_seeds,
    run_policy_episode,
)
from sim import SimEnv


EvalSelectionMode = Literal["mode-a", "mode-b"]


GRASP_SCORE_WEIGHT = 0.10
LIFT_SCORE_WEIGHT = 0.10
TRAY_REACH_SCORE_WEIGHT = 0.10
LOWERED_SCORE_WEIGHT = 0.10
RELEASED_SCORE_WEIGHT = 0.10
PLACEMENT_SCORE_WEIGHT = 0.50
EVAL_METRIC_VERSION = 4
EVAL_SELECTION_MODES: tuple[EvalSelectionMode, ...] = ("mode-a", "mode-b")
DEFAULT_EVAL_SELECTION_MODE: EvalSelectionMode = "mode-a"


class ScoreResult(TypedDict):
    eval_metric_version: int
    scores: list[float]
    mean_score: float
    grasp_rate: float
    lift_rate: float
    tray_reach_rate: float
    lowered_to_tray_rate: float
    released_over_tray_rate: float
    placement_success_rate: float


_eval_worker_state = None


def eval_selection_key(
    *,
    selection_mode: EvalSelectionMode,
    mean_score: float,
    placement_success_rate: float,
) -> tuple[float, ...]:
    if selection_mode == "mode-a":
        return (mean_score,)
    if selection_mode == "mode-b":
        return (placement_success_rate, mean_score)
    raise ValueError(f"Unknown evaluation selection mode: {selection_mode}")


def score_task_metrics(*, metrics: TaskMetrics) -> float:
    return (
        GRASP_SCORE_WEIGHT * float(metrics["grasped"])
        + LIFT_SCORE_WEIGHT * float(metrics["lifted"])
        + TRAY_REACH_SCORE_WEIGHT * float(metrics["tray_reached"])
        + LOWERED_SCORE_WEIGHT * float(metrics["lowered_to_tray"])
        + RELEASED_SCORE_WEIGHT * float(metrics["released_over_tray"])
        + PLACEMENT_SCORE_WEIGHT * float(metrics["placement_success"])
    )


def summarize_task_metrics(*, episode_metrics: list[TaskMetrics]) -> ScoreResult:
    scores = [score_task_metrics(metrics=metrics) for metrics in episode_metrics]
    return {
        "eval_metric_version": EVAL_METRIC_VERSION,
        "scores": scores,
        "mean_score": float(np.mean(scores)),
        "grasp_rate": float(np.mean([item["grasped"] for item in episode_metrics])),
        "lift_rate": float(np.mean([item["lifted"] for item in episode_metrics])),
        "tray_reach_rate": float(
            np.mean([item["tray_reached"] for item in episode_metrics])
        ),
        "lowered_to_tray_rate": float(
            np.mean([item["lowered_to_tray"] for item in episode_metrics])
        ),
        "released_over_tray_rate": float(
            np.mean([item["released_over_tray"] for item in episode_metrics])
        ),
        "placement_success_rate": float(
            np.mean([item["placement_success"] for item in episode_metrics])
        ),
    }


def initialize_eval_worker(ckpt_path: str) -> None:
    global _eval_worker_state

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    model, normalize, action_space, norm_dict = load_policy(
        model_path=Path(ckpt_path),
        device=device,
    )
    sim = SimEnv(randomize_scene=True)
    _eval_worker_state = (
        sim,
        model,
        device,
        normalize,
        action_space,
        norm_dict,
    )


def score_episode_worker(task: tuple[int, int, bool]) -> TaskMetrics:
    if _eval_worker_state is None:
        raise RuntimeError("Evaluation worker was not initialized")

    seed, max_steps, expert_baseline = task
    sim, model, device, normalize, action_space, norm_dict = _eval_worker_state
    sim.rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed)

    with torch.inference_mode():
        result = run_policy_episode(
            sim=sim,
            model=model,
            device=device,
            normalize=normalize,
            action_space=action_space,
            norm_dict=norm_dict,
            max_steps=max_steps,
            track_phase=True,
            dagger=expert_baseline,
            beta=1.0 if expert_baseline else 0.0,
            rng=rng,
        )
    return result["task_metrics"]


def score_ckpt(
    *,
    ckpt_path: str,
    seed: int,
    max_steps: int,
    episodes: int,
    expert_baseline: bool = False,
    workers: int = 1,
) -> ScoreResult:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    episode_seeds = make_episode_seeds(seed=seed, episodes=episodes)
    tasks = [
        (episode_seed, max_steps, expert_baseline)
        for episode_seed in episode_seeds
    ]

    if workers > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, episodes),
            mp_context=context,
            initializer=initialize_eval_worker,
            initargs=(ckpt_path,),
        ) as executor:
            episode_metrics = list(executor.map(score_episode_worker, tasks))
        return summarize_task_metrics(episode_metrics=episode_metrics)

    device = torch.device("cpu")
    model, normalize, action_space, norm_dict = load_policy(
        model_path=Path(ckpt_path),
        device=device,
    )
    sim = SimEnv(randomize_scene=True)
    episode_metrics = []
    with torch.inference_mode():
        for episode_seed in episode_seeds:
            sim.rng = np.random.default_rng(episode_seed)
            result = run_policy_episode(
                sim=sim,
                model=model,
                device=device,
                normalize=normalize,
                action_space=action_space,
                norm_dict=norm_dict,
                max_steps=max_steps,
                track_phase=True,
                dagger=expert_baseline,
                beta=1.0 if expert_baseline else 0.0,
                rng=np.random.default_rng(episode_seed),
            )
            episode_metrics.append(result["task_metrics"])

    return summarize_task_metrics(episode_metrics=episode_metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=1400)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--expert-baseline", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    score_dict = score_ckpt(
        ckpt_path=args.ckpt_path,
        seed=args.seed,
        max_steps=args.max_steps,
        episodes=args.episodes,
        expert_baseline=args.expert_baseline,
        workers=args.workers,
    )
    print(f"Mean score: {score_dict['mean_score']:.4f}")
    print(f"Grasp rate: {score_dict['grasp_rate']:.4f}")
    print(f"Lift rate: {score_dict['lift_rate']:.4f}")
    print(f"Tray reach rate: {score_dict['tray_reach_rate']:.4f}")
    print(f"Lowered-to-tray rate: {score_dict['lowered_to_tray_rate']:.4f}")
    print(f"Released-over-tray rate: {score_dict['released_over_tray_rate']:.4f}")
    print(
        "Placement success rate: "
        f"{score_dict['placement_success_rate']:.4f}"
    )
    print(f"Scores: {score_dict['scores']}")


if __name__ == "__main__":
    main()
