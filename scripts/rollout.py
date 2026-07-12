import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import mujoco
import mujoco.viewer
import numpy as np
import torch
from tqdm import tqdm

from constant import (
    ACTION_DIMS,
    EPSILON,
    MAX_ARM_DELTA,
)
from expert import Phase, PickPlaceController
from mlp import MLP
from sim import SimEnv


DEFAULT_DAGGER_DIR = Path("data/dagger")


class NormDict(TypedDict):
    arm_actions_mean: torch.Tensor
    arm_actions_std: torch.Tensor
    arm_obs_mean: torch.Tensor
    arm_obs_std: torch.Tensor


class EpisodeResult(TypedDict):
    steps: int
    cube_init_pos: np.ndarray
    tray_init_pos: np.ndarray
    log_buffers: dict[str, list[np.ndarray]]
    observations: list[np.ndarray]
    expert_actions: list[np.ndarray]
    policy_actions: list[np.ndarray]
    executed_actions: list[np.ndarray]
    expert_action_mask: list[bool]


def load_policy(
    *,
    model_path: Path,
    device: torch.device,) -> tuple[MLP, bool, str, NormDict]:
    model = MLP().to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_dict"])
    model.eval()

    norm_dict: NormDict = {
        key: torch.from_numpy(checkpoint[key]).to(device).squeeze(0)
        for key in (
            "arm_actions_mean",
            "arm_actions_std",
            "arm_obs_mean",
            "arm_obs_std",
        )
    }
    return model, checkpoint["normalize"], checkpoint["action_space"], norm_dict


def make_rollout_log_dir(*, log_root: Path, model_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_name = model_path.parent.name or model_path.stem
    return log_root / f"{timestamp}_{checkpoint_name}"


def make_dagger_log_dir(*, dagger_root: Path, model_path: Path, beta: float) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_name = model_path.parent.name or model_path.stem
    beta_name = f"beta{beta:.2f}".replace(".", "p")
    return dagger_root / f"{timestamp}_{checkpoint_name}_{beta_name}"


def prepare_rollout_log_dir(
    *,
    enabled: bool,
    log_root: Path,
    model_path: Path,
    train_npz_dir: Path,) -> Path | None:
    if not enabled:
        return None
    if not train_npz_dir.exists():
        raise FileNotFoundError(f"Training npz directory not found: {train_npz_dir}")

    log_dir = make_rollout_log_dir(log_root=log_root, model_path=model_path)
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"Rollout log dir: {log_dir}")
    return log_dir


def prepare_dagger_dir(
    *,
    enabled: bool,
    dagger_root: Path,
    model_path: Path,
    beta: float,
    create_subdir: bool = True,) -> Path | None:
    if not enabled:
        return None

    if not create_subdir:
        dagger_root.mkdir(parents=True, exist_ok=True)
        print(f"Dagger log dir: {dagger_root}")
        return dagger_root

    dagger_dir = make_dagger_log_dir(
        dagger_root=dagger_root,
        model_path=model_path,
        beta=beta,
    )
    dagger_dir.mkdir(parents=True, exist_ok=False)
    print(f"Dagger log dir: {dagger_dir}")
    return dagger_dir


def append_step_log(
    *,
    buffers: dict[str, list[np.ndarray]],
    obs: torch.Tensor,
    actions: torch.Tensor,
    executed_actions: np.ndarray,
    joints_pred_raw: torch.Tensor,
    joints_pred_unnorm: torch.Tensor,) -> None:
    buffers["rollout_obs"].append(obs.squeeze(0).detach().cpu().numpy())
    buffers["rollout_actions"].append(actions.squeeze(0).detach().cpu().numpy())
    buffers["executed_actions"].append(executed_actions.copy())
    buffers["joints_pred_raw"].append(joints_pred_raw.squeeze(0).detach().cpu().numpy())
    buffers["joints_pred_unnorm"].append(
        joints_pred_unnorm.squeeze(0).detach().cpu().numpy()
    )


def save_episode_log(
    *,
    log_dir: Path,
    train_npz_dir: Path,
    episode_idx: int,
    action_space: str,
    buffers: dict[str, list[np.ndarray]],) -> None:
    train_episode_path = train_npz_dir / f"pick_place_{episode_idx:06d}.npz"
    if not train_episode_path.exists():
        raise FileNotFoundError(
            f"Training episode file not found: {train_episode_path}"
        )

    with np.load(train_episode_path) as data:
        missing_keys = {"obs", "actions"} - set(data.files)
        if missing_keys:
            raise KeyError(
                f"{train_episode_path} is missing keys: {sorted(missing_keys)}"
            )
        train_obs = np.asarray(data["obs"], dtype=np.float32)
        train_actions = np.asarray(data["actions"], dtype=np.float32)

    rollout_obs = np.asarray(buffers["rollout_obs"], dtype=np.float32)
    rollout_actions = np.asarray(buffers["rollout_actions"], dtype=np.float32)
    executed_actions = np.asarray(buffers["executed_actions"], dtype=np.float32)
    joints_pred_raw = np.asarray(buffers["joints_pred_raw"], dtype=np.float32)
    joints_pred_unnorm = np.asarray(buffers["joints_pred_unnorm"], dtype=np.float32)

    arrays = {
        "rollout_obs": rollout_obs,
        "train_obs": train_obs,
        "rollout_actions": rollout_actions,
        "executed_actions": executed_actions,
        "train_actions": train_actions,
        "joints_pred_raw": joints_pred_raw,
        "joints_pred_unnorm": joints_pred_unnorm,
    }
    if "dagger_obs" in buffers and "dagger_actions" in buffers:
        dagger_obs = np.asarray(buffers["dagger_obs"], dtype=np.float32)
        dagger_actions = np.asarray(buffers["dagger_actions"], dtype=np.float32)
        arrays["dagger_obs"] = dagger_obs
        arrays["dagger_actions"] = dagger_actions

    if action_space == "joint_delta":
        arrays["rollout_action_deltas"] = (
            rollout_actions[:, : ACTION_DIMS - 1] - rollout_obs[:, : ACTION_DIMS - 1]
        )
        arrays["executed_action_deltas"] = (
            executed_actions[:, : ACTION_DIMS - 1] - rollout_obs[:, : ACTION_DIMS - 1]
        )
        arrays["train_action_deltas"] = (
            train_actions[:, : ACTION_DIMS - 1] - train_obs[:, : ACTION_DIMS - 1]
        )
        if "dagger_obs" in arrays and "dagger_actions" in arrays:
            arrays["dagger_action_deltas"] = (
                arrays["dagger_actions"][:, : ACTION_DIMS - 1]
                - arrays["dagger_obs"][:, : ACTION_DIMS - 1]
            )

    episode_path = log_dir / f"episode_{episode_idx:06d}.npz"
    np.savez_compressed(episode_path, **arrays)


def save_dagger_episode(
    *,
    dagger_dir: Path,
    episode_idx: int,
    beta: float,
    result: EpisodeResult,) -> None:
    dagger_path = dagger_dir / f"pick_place_{episode_idx:06d}.npz"
    np.savez_compressed(
        dagger_path,
        obs=np.asarray(result["observations"], dtype=np.float32),
        actions=np.asarray(result["expert_actions"], dtype=np.float32),
        policy_actions=np.asarray(result["policy_actions"], dtype=np.float32),
        executed_actions=np.asarray(result["executed_actions"], dtype=np.float32),
        execute_expert=np.asarray(result["expert_action_mask"], dtype=np.bool_),
        beta=np.asarray(beta, dtype=np.float32),
        cube_init_pos=np.asarray(result["cube_init_pos"], dtype=np.float32),
        tray_init_pos=np.asarray(result["tray_init_pos"], dtype=np.float32),
    )


def predict_policy_action(
    *,
    model: MLP,
    obs: np.ndarray,
    device: torch.device,
    normalize: bool,
    action_space: str,
    norm_dict: NormDict,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict and post-process a simulator-ready policy action."""
    obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0)
    obs_target = obs_tensor
    if normalize:
        obs_target = (obs_tensor - norm_dict["arm_obs_mean"]) / (
            norm_dict["arm_obs_std"] + EPSILON
        )

    joints_pred, gripper_pred = model(obs_target)
    joints_pred_unnorm = joints_pred
    if normalize:
        joints_pred_unnorm = (
            joints_pred * (norm_dict["arm_actions_std"] + EPSILON)
            + norm_dict["arm_actions_mean"]
        )

    joints_actions = joints_pred_unnorm
    if action_space == "joint_delta":
        joints_actions = torch.clamp(
            joints_actions,
            min=-MAX_ARM_DELTA,
            max=MAX_ARM_DELTA,
        )
        joints_actions = joints_actions + obs_tensor[:, : ACTION_DIMS - 1]

    gripper_prob = torch.sigmoid(gripper_pred)
    gripper_actions = torch.where(gripper_prob >= 0.5, 255.0, 0.0)
    policy_actions = torch.concat([joints_actions, gripper_actions], dim=1)
    return obs_tensor, policy_actions, joints_pred, joints_pred_unnorm


def run_policy_episode(
    *,
    sim: SimEnv,
    model: MLP,
    device: torch.device,
    normalize: bool = True,
    action_space: str = "joint_delta",
    norm_dict: NormDict,
    max_steps: int,
    track_phase: bool = False,
    dagger: bool = False,
    beta: float = 0.0,
    rng: np.random.Generator,
    log_rollout: bool = False,
    viewer=None,
    should_stop=None,
    phase_callback=None,) -> EpisodeResult:
    """Run one policy episode, stopping when the controller reaches DONE."""
    observations: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    policy_action_history: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    expert_action_mask: list[bool] = []

    sim.reset_episode()
    cube_init_pos = sim.data.qpos[sim.cube_qpos_addr : sim.cube_qpos_addr + 3].copy()
    tray_init_pos = sim.data.site("tray_center").xpos.copy()

    controller = (
        PickPlaceController(sim.model, sim.data) if track_phase or dagger else None
    )
    if controller is not None and phase_callback is not None:
        phase_callback(0, controller.phase)

    log_buffers: dict[str, list[np.ndarray]] = {}
    if log_rollout:
        log_buffers = {
            "rollout_obs": [],
            "rollout_actions": [],
            "executed_actions": [],
            "joints_pred_raw": [],
            "joints_pred_unnorm": [],
        }
        if dagger:
            log_buffers["dagger_obs"] = observations
            log_buffers["dagger_actions"] = expert_actions

    steps = 0
    while (
        (should_stop is None or not should_stop())
        and steps < max_steps
        and (viewer is None or viewer.is_running())
        and (controller is None or controller.phase != Phase.DONE)
    ):
        obs = sim.build_observation()
        if dagger:
            observations.append(obs.copy())

        obs_tensor, policy_actions, joints_pred, joints_pred_unnorm = (
            predict_policy_action(
                model=model,
                obs=obs,
                device=device,
                normalize=normalize,
                action_space=action_space,
                norm_dict=norm_dict,
            )
        )
        policy_action_np = policy_actions.squeeze(0).detach().cpu().numpy()

        execute_expert_action = False
        expert_action = None
        if dagger and controller is not None:
            expert_action = controller.compute_actions()
            expert_actions.append(expert_action)
            execute_expert_action = rng.random() < beta

        action_to_execute = expert_action if execute_expert_action else policy_action_np
        if dagger:
            policy_action_history.append(policy_action_np.copy())
            executed_actions.append(action_to_execute.copy())
            expert_action_mask.append(execute_expert_action)

        if log_rollout:
            append_step_log(
                buffers=log_buffers,
                obs=obs_tensor,
                actions=policy_actions,
                executed_actions=action_to_execute,
                joints_pred_raw=joints_pred,
                joints_pred_unnorm=joints_pred_unnorm,
            )

        sim.data.ctrl[: sim.model.nu] = action_to_execute
        mujoco.mj_step(sim.model, sim.data)
        if controller is not None:
            previous_phase = controller.phase
            controller.update_phase()
            if controller.phase != previous_phase and phase_callback is not None:
                phase_callback(steps, controller.phase)

        if viewer is not None:
            time.sleep(sim.model.opt.timestep * 2)
            viewer.sync()
        steps += 1

    if not (
        len(observations)
        == len(expert_actions)
        == len(policy_action_history)
        == len(executed_actions)
        == len(expert_action_mask)
    ):
        raise ValueError("Dagger episode buffers have mismatched lengths")

    return {
        "steps": steps,
        "cube_init_pos": cube_init_pos,
        "tray_init_pos": tray_init_pos,
        "log_buffers": log_buffers,
        "observations": observations,
        "expert_actions": expert_actions,
        "policy_actions": policy_action_history,
        "executed_actions": executed_actions,
        "expert_action_mask": expert_action_mask,
    }


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
    create_dagger_subdir: bool = True,
    headless: bool,):
    # sim = SimEnv()
    sim = SimEnv(randomize_scene=randomize_scene, seed=seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(model_path)
    model, normalize, action_space, norm_dict = load_policy(
        model_path=model_path,
        device=device,
    )

    if not normalize:
        raise ValueError("Checkpoint must use normalized observations and actions")
    if action_space not in ("joint_delta", "absolute"):
        raise ValueError(f"Checkpoint has unsupported action space: {action_space!r}")

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
        create_subdir=create_dagger_subdir,
    )

    quit_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal quit_requested
        if chr(keycode).lower() == "q":
            quit_requested = True

    def run_rollouts(*, viewer=None) -> None:
        with torch.no_grad():
            progress_is_tty = sys.stderr.isatty()
            episode_pbar = tqdm(
                range(episodes),
                desc="rollout",
                unit="episode",
                disable=not progress_is_tty,
            )

            def update_phase_progress(step, phase) -> None:
                if progress_is_tty:
                    episode_pbar.set_postfix_str(f"step={step} phase={phase.name}")

            for episode in episode_pbar:
                result = run_policy_episode(
                    sim=sim,
                    model=model,
                    device=device,
                    normalize=normalize,
                    action_space=action_space,
                    norm_dict=norm_dict,
                    max_steps=max_steps,
                    track_phase=dagger,
                    dagger=dagger,
                    beta=beta,
                    rng=rng,
                    log_rollout=log_rollout and log_dir is not None,
                    viewer=viewer,
                    should_stop=lambda: quit_requested,
                    phase_callback=update_phase_progress,
                )

                if log_rollout and log_dir is not None:
                    save_episode_log(
                        log_dir=log_dir,
                        train_npz_dir=train_npz_dir,
                        episode_idx=episode,
                        action_space=action_space,
                        buffers=result["log_buffers"],
                    )

                if dagger and dagger_dir is not None:
                    save_dagger_episode(
                        dagger_dir=dagger_dir,
                        episode_idx=episode,
                        beta=beta,
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
        "--headless", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()
    if not 0.0 <= args.beta <= 1.0:
        parser.error(f"--beta must be in [0.0, 1.0], got {args.beta}")
    if args.beta > 0.0 and not args.dagger:
        parser.error("--beta requires --dagger")

    return args


def main():
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
        create_dagger_subdir=True,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
