"""Run policy and DAgger rollouts.

Examples using the viewer and saving DAgger samples:

Beta mode randomly executes the expert with the given probability::

    uv run scripts/rollout.py --model checkpoints/run/best.pt \
        --dagger --dagger-mode beta --beta 0.5 --no-log-rollout

Threshold mode starts a fixed expert burst when the L2 arm-action
disagreement exceeds the threshold while no burst is active::

    uv run scripts/rollout.py --model checkpoints/run/best.pt \
        --dagger --dagger-mode threshold --intervention-threshold 0.2 \
        --intervention-steps 50 --no-log-rollout

The viewer is enabled unless ``--headless`` is provided.
"""

import argparse
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from tqdm import tqdm

# These constants remain available for callers that historically imported them
# from rollout.py.
from constant import (
    DEFAULT_CAPTURE_HZ,
    LOWERING_STABLE_STEPS,
    PLACEMENT_STABLE_STEPS,
    RELEASE_STABLE_STEPS,
    RETREAT_STABLE_STEPS,
)
from expert import Phase
from policy_runtimes.registry import load_runtime
from policy_runtimes.types import PolicyRuntime
from rollout_core.dagger import (
    DAGGER_INTERVENTION_MODES,
    DEFAULT_INTERVENTION_STEPS,
)
from rollout_core.episode import EpisodeResult, run_policy_episode
from rollout_core.metrics import TaskMetrics, TaskMetricsTracker
from rollout_core.persistence import (
    append_step_log,
    make_dagger_log_dir,
    make_rollout_log_dir,
    prepare_dagger_dir,
    prepare_rollout_log_dir,
    save_dagger_episode,
    save_episode_log,
)
from sim import SimEnv


DEFAULT_DAGGER_DIR = Path("data/dagger")


_dagger_worker_state: tuple[SimEnv, PolicyRuntime] | None = None


def make_episode_seeds(*, seed: int, episodes: int) -> list[int]:
    seed_sequence = np.random.SeedSequence(seed)
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(episodes)
    ]


def initialize_dagger_worker(config: tuple[str, bool]) -> None:
    global _dagger_worker_state

    model_path, randomize_scene = config
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    runtime = load_runtime(
        model_path=Path(model_path),
        device=device,
    )
    sim = SimEnv(randomize_scene=randomize_scene)
    _dagger_worker_state = (sim, runtime)


def run_dagger_worker(
    task: tuple[int, int, int, float, str, str, float | None, int, float],
) -> tuple[int, int]:
    if _dagger_worker_state is None:
        raise RuntimeError("DAgger worker was not initialized")

    (
        episode_idx,
        seed,
        max_steps,
        beta,
        dagger_dir,
        intervention_mode,
        intervention_threshold,
        intervention_steps,
        capture_hz,
    ) = task
    sim, runtime = _dagger_worker_state
    sim.rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed)

    with torch.inference_mode():
        result = run_policy_episode(
            sim=sim,
            runtime=runtime,
            max_steps=max_steps,
            track_phase=True,
            dagger=True,
            beta=beta,
            intervention_mode=intervention_mode,
            intervention_threshold=intervention_threshold,
            intervention_steps=intervention_steps,
            capture_hz=capture_hz,
            rng=rng,
            episode_seed=seed,
        )
    save_dagger_episode(
        dagger_dir=Path(dagger_dir),
        episode_idx=episode_idx,
        beta=beta,
        intervention_mode=intervention_mode,
        intervention_threshold=intervention_threshold,
        intervention_steps=intervention_steps,
        result=result,
    )
    return episode_idx, result["steps"]


def rollout(
    *,
    model_path: str,
    randomize_scene: bool,
    seed: int,
    episodes: int,
    max_steps: int,
    log_root: str | Path | None,
    train_npz_dir: str | Path | None,
    log_rollout: bool,
    dagger: bool,
    dagger_root: str | Path,
    beta: float,
    intervention_mode: str = "beta",
    intervention_threshold: float | None = None,
    intervention_steps: int = DEFAULT_INTERVENTION_STEPS,
    create_dagger_subdir: bool = True,
    headless: bool,
    workers: int = 1,
    capture_hz: float = DEFAULT_CAPTURE_HZ,
) -> None:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if intervention_mode not in DAGGER_INTERVENTION_MODES:
        raise ValueError(
            f"Unsupported DAgger intervention mode: {intervention_mode!r}"
        )
    if intervention_mode == "threshold" and intervention_threshold is None:
        raise ValueError("Threshold intervention mode requires a threshold")
    if intervention_threshold is not None and intervention_threshold < 0.0:
        raise ValueError("intervention threshold must be non-negative")
    if intervention_steps < 1:
        raise ValueError("intervention steps must be at least 1")
    if capture_hz <= 0.0:
        raise ValueError("capture_hz must be positive")

    if workers > 1:
        if not headless or not dagger or log_rollout:
            raise ValueError(
                "parallel rollouts require headless DAgger with rollout logging disabled"
            )

        model_path = Path(model_path)
        dagger_dir = prepare_dagger_dir(
            enabled=True,
            dagger_root=Path(dagger_root),
            model_path=model_path,
            beta=beta,
            intervention_mode=intervention_mode,
            intervention_threshold=intervention_threshold,
            create_subdir=create_dagger_subdir,
        )
        if dagger_dir is None:
            raise RuntimeError("DAgger output directory was not created")

        episode_seeds = make_episode_seeds(seed=seed, episodes=episodes)
        tasks = [
            (
                episode_idx,
                episode_seed,
                max_steps,
                beta,
                str(dagger_dir),
                intervention_mode,
                intervention_threshold,
                intervention_steps,
                capture_hz,
            )
            for episode_idx, episode_seed in enumerate(episode_seeds)
        ]
        worker_count = min(workers, episodes)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=initialize_dagger_worker,
            initargs=((str(model_path), randomize_scene),),
        ) as executor:
            list(executor.map(run_dagger_worker, tasks))
        return

    sim = SimEnv(randomize_scene=randomize_scene, seed=seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(model_path)
    runtime = load_runtime(
        model_path=model_path,
        device=device,
    )

    if not runtime.normalize:
        raise ValueError("Checkpoint must use normalized observations and actions")
    if runtime.action_space not in ("joint_delta", "absolute"):
        raise ValueError(
            f"Checkpoint has unsupported action space: {runtime.action_space!r}"
        )

    log_dir = None
    if log_rollout:
        if log_root is None:
            raise ValueError("log_root is required when log_rollout is enabled")
        if train_npz_dir is None:
            raise ValueError("train_npz_dir is required when log_rollout is enabled")
        train_npz_dir = Path(train_npz_dir)
        log_dir = prepare_rollout_log_dir(
            enabled=True,
            log_root=Path(log_root),
            model_path=model_path,
            train_npz_dir=train_npz_dir,
        )
    dagger_dir = prepare_dagger_dir(
        enabled=dagger,
        dagger_root=Path(dagger_root),
        model_path=model_path,
        beta=beta,
        intervention_mode=intervention_mode,
        intervention_threshold=intervention_threshold,
        create_subdir=create_dagger_subdir,
    )

    quit_requested = False
    advance_episode_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal quit_requested, advance_episode_requested
        key = chr(keycode).lower()
        if key == "q":
            quit_requested = True
        elif key == "n" and not dagger:
            advance_episode_requested = True

    def run_rollouts(*, viewer=None) -> None:
        nonlocal advance_episode_requested
        with torch.no_grad():
            progress_is_tty = sys.stderr.isatty()
            episode_pbar = tqdm(
                range(episodes),
                desc="rollout",
                unit="episode",
                disable=not progress_is_tty,
            )

            current_phase = None
            current_step = 0
            expert_was_executing = False
            last_expert_sound_at = float("-inf")

            def update_progress() -> None:
                if not progress_is_tty:
                    return
                phase_name = current_phase.name if current_phase is not None else "-"
                control_name = "expert" if expert_was_executing else "policy"
                episode_pbar.set_postfix_str(
                    f"step={current_step} phase={phase_name} control={control_name}"
                )

            def update_phase_progress(step, phase) -> None:
                nonlocal current_phase, current_step
                current_step = step
                current_phase = phase
                update_progress()

            def update_control_progress(step, execute_expert) -> None:
                nonlocal current_step, expert_was_executing, last_expert_sound_at
                current_step = step
                now = time.monotonic()
                if (
                    progress_is_tty
                    and execute_expert
                    and not expert_was_executing
                    and now - last_expert_sound_at >= 0.25
                ):
                    try:
                        subprocess.Popen(
                            ["canberra-gtk-play", "--id=bell"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except OSError:
                        sys.stderr.write("\a")
                        sys.stderr.flush()
                    last_expert_sound_at = now
                expert_was_executing = execute_expert
                update_progress()

            episode_seeds = (
                make_episode_seeds(seed=seed, episodes=episodes) if dagger else []
            )
            for episode in episode_pbar:
                advance_episode_requested = False
                current_phase = None
                current_step = 0
                expert_was_executing = False
                last_expert_sound_at = float("-inf")
                episode_seed = episode_seeds[episode] if episode_seeds else None
                if episode_seed is not None:
                    sim.rng = np.random.default_rng(episode_seed)
                result = run_policy_episode(
                    sim=sim,
                    runtime=runtime,
                    max_steps=max_steps,
                    track_phase=dagger,
                    dagger=dagger,
                    beta=beta,
                    intervention_mode=intervention_mode,
                    intervention_threshold=intervention_threshold,
                    intervention_steps=intervention_steps,
                    capture_hz=capture_hz,
                    rng=(
                        np.random.default_rng(episode_seed)
                        if episode_seed is not None
                        else rng
                    ),
                    episode_seed=episode_seed,
                    log_rollout=log_rollout and log_dir is not None,
                    viewer=viewer,
                    should_stop=lambda: quit_requested or advance_episode_requested,
                    phase_callback=update_phase_progress,
                    control_callback=update_control_progress,
                )

                if log_rollout and log_dir is not None:
                    save_episode_log(
                        log_dir=log_dir,
                        train_npz_dir=train_npz_dir,
                        episode_idx=episode,
                        action_space=runtime.action_space,
                        buffers=result["log_buffers"],
                    )

                if dagger and dagger_dir is not None:
                    save_dagger_episode(
                        dagger_dir=dagger_dir,
                        episode_idx=episode,
                        beta=beta,
                        intervention_mode=intervention_mode,
                        intervention_threshold=intervention_threshold,
                        intervention_steps=intervention_steps,
                        result=result,
                    )

    if headless:
        run_rollouts()
    else:
        with mujoco.viewer.launch_passive(
            sim.model,
            sim.data,
            key_callback=key_callback,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = (0.45, 0.0, 0.78)
            viewer.cam.distance = 1.35
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -25
            run_rollouts(viewer=viewer)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model path e.g. checkpoints/026_mlp_action_norm",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--randomize-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs/rollouts"))
    parser.add_argument(
        "--train-npz-dir", type=Path, default=Path("data/train/rand-100")
    )
    parser.add_argument(
        "--log-rollout", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dagger", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--dagger-dir", type=Path, default=DEFAULT_DAGGER_DIR)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--dagger-mode",
        choices=DAGGER_INTERVENTION_MODES,
        default="beta",
        help="Choose random beta mixing or disagreement-triggered interventions",
    )
    parser.add_argument(
        "--intervention-threshold",
        type=float,
        default=None,
        help="L2 arm-action disagreement that triggers threshold intervention",
    )
    parser.add_argument(
        "--intervention-steps",
        type=int,
        default=DEFAULT_INTERVENTION_STEPS,
        help="Minimum expert burst after disagreement exceeds the threshold",
    )
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--capture-hz",
        type=float,
        default=DEFAULT_CAPTURE_HZ,
        help="Policy inference rate for aligned obs, action, and image samples.",
    )

    args = parser.parse_args()
    if not 0.0 <= args.beta <= 1.0:
        parser.error(f"--beta must be in [0.0, 1.0], got {args.beta}")
    if args.beta > 0.0 and not args.dagger:
        parser.error("--beta requires --dagger")
    if args.dagger_mode != "beta" and not args.dagger:
        parser.error("--dagger-mode requires --dagger")
    if args.dagger_mode == "threshold":
        if args.intervention_threshold is None:
            parser.error("threshold mode requires --intervention-threshold")
        if args.beta != 0.0:
            parser.error("--beta cannot be used with threshold mode")
    if args.intervention_threshold is not None and args.intervention_threshold < 0.0:
        parser.error("--intervention-threshold must be non-negative")
    if args.intervention_steps < 1:
        parser.error("--intervention-steps must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.capture_hz <= 0.0:
        parser.error("--capture-hz must be positive")

    return args


def main() -> None:
    args = parse_args()
    rollout(
        model_path=args.model,
        randomize_scene=args.randomize_scene,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        log_root=args.log_dir,
        train_npz_dir=args.train_npz_dir,
        log_rollout=args.log_rollout,
        dagger=args.dagger,
        dagger_root=args.dagger_dir,
        beta=args.beta,
        intervention_mode=args.dagger_mode,
        intervention_threshold=args.intervention_threshold,
        intervention_steps=args.intervention_steps,
        create_dagger_subdir=True,
        headless=args.headless,
        workers=args.workers,
        capture_hz=args.capture_hz,
    )


if __name__ == "__main__":
    main()
