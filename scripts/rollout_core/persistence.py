"""Rollout output naming, buffering, and NPZ persistence."""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from constant import ACTION_DIMS
from rollout_core.dagger import DEFAULT_INTERVENTION_STEPS

if TYPE_CHECKING:
    from rollout_core.episode import EpisodeResult


def make_rollout_log_dir(*, log_root: Path, model_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_name = model_path.parent.name or model_path.stem
    return log_root / f"{timestamp}_{checkpoint_name}"


def make_dagger_log_dir(
    *,
    dagger_root: Path,
    model_path: Path,
    beta: float,
    intervention_mode: str,
    intervention_threshold: float | None,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_name = model_path.parent.name or model_path.stem
    if intervention_mode == "threshold":
        if intervention_threshold is None:
            raise ValueError("Threshold intervention mode requires a threshold")
        intervention_name = f"threshold{intervention_threshold:g}".replace(".", "p")
    else:
        intervention_name = f"beta{beta:.2f}".replace(".", "p")
    return dagger_root / f"{timestamp}_{checkpoint_name}_{intervention_name}"


def prepare_rollout_log_dir(
    *,
    enabled: bool,
    log_root: Path,
    model_path: Path,
    train_npz_dir: Path,
) -> Path | None:
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
    intervention_mode: str,
    intervention_threshold: float | None,
    create_subdir: bool = True,
) -> Path | None:
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
        intervention_mode=intervention_mode,
        intervention_threshold=intervention_threshold,
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
    joints_pred_unnorm: torch.Tensor,
) -> None:
    buffers["rollout_obs"].append(obs.squeeze(0).detach().cpu().numpy())
    buffers["rollout_actions"].append(actions.squeeze(0).detach().cpu().numpy())
    buffers["executed_actions"].append(executed_actions.copy())
    buffers["joints_pred_raw"].append(
        joints_pred_raw.squeeze(0).detach().cpu().numpy()
    )
    buffers["joints_pred_unnorm"].append(
        joints_pred_unnorm.squeeze(0).detach().cpu().numpy()
    )


def save_episode_log(
    *,
    log_dir: Path,
    train_npz_dir: Path,
    episode_idx: int,
    action_space: str,
    buffers: dict[str, list[np.ndarray]],
) -> None:
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
    joints_pred_unnorm = np.asarray(
        buffers["joints_pred_unnorm"], dtype=np.float32
    )

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
            executed_actions[:, : ACTION_DIMS - 1]
            - rollout_obs[:, : ACTION_DIMS - 1]
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
    intervention_mode: str = "beta",
    intervention_threshold: float | None = None,
    intervention_steps: int = DEFAULT_INTERVENTION_STEPS,
    result: "EpisodeResult",
) -> None:
    dagger_path = dagger_dir / f"pick_place_{episode_idx:06d}.npz"
    arrays = {
        "obs": np.asarray(result["observations"], dtype=np.float32),
        "actions": np.asarray(result["expert_actions"], dtype=np.float32),
        "policy_actions": np.asarray(result["policy_actions"], dtype=np.float32),
        "executed_actions": np.asarray(result["executed_actions"], dtype=np.float32),
        "execute_expert": np.asarray(result["expert_action_mask"], dtype=np.bool_),
        "phases": np.asarray(result["phases"], dtype=np.int8),
        "arm_disagreement": np.asarray(result["arm_disagreement"], dtype=np.float32),
        "gripper_disagreement": np.asarray(
            result["gripper_disagreement"], dtype=np.bool_
        ),
        "terminal_reason": np.asarray(result["terminal_reason"]),
        "final_phase": np.asarray(result["final_phase"], dtype=np.int8),
        "seed": np.asarray(result["seed"], dtype=np.uint32),
        "steps": np.asarray(result["steps"], dtype=np.int32),
        **{
            name: np.asarray(value, dtype=np.bool_)
            for name, value in result["task_metrics"].items()
        },
        "beta": np.asarray(beta, dtype=np.float32),
        "intervention_mode": np.asarray(intervention_mode),
        "intervention_threshold": np.asarray(
            np.nan if intervention_threshold is None else intervention_threshold,
            dtype=np.float32,
        ),
        "intervention_steps": np.asarray(intervention_steps, dtype=np.int32),
        "cube_init_pos": np.asarray(result["cube_init_pos"], dtype=np.float32),
        "tray_init_pos": np.asarray(result["tray_init_pos"], dtype=np.float32),
    }
    if result["img_observations"]:
        arrays["img_obs"] = np.asarray(result["img_observations"], dtype=np.uint8)
    np.savez_compressed(dagger_path, **arrays)
