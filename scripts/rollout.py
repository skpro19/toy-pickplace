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
from expert import (
    CUBE_LIFT_MIN_DELTA,
    TRAY_PLACE_TOL,
    Phase,
    PickPlaceController,
)
from mlp import MLP
from sim import SimEnv


DEFAULT_DAGGER_DIR = Path("data/dagger")
DAGGER_INTERVENTION_MODES = ("beta", "threshold")
DEFAULT_INTERVENTION_STEPS = 50
GRASP_STABLE_STEPS = 5
LOWERING_STABLE_STEPS = 5
RELEASE_STABLE_STEPS = 5
PLACEMENT_STABLE_STEPS = 50
PLACEMENT_MAX_SPEED = 0.05
PLACEMENT_Z_TOL = 0.04
TRAY_INNER_XY_TOL = 0.05


_dagger_worker_state: tuple[
    SimEnv,
    MLP,
    torch.device,
    bool,
    str,
    "NormDict",
] | None = None


class NormDict(TypedDict):
    arm_actions_mean: torch.Tensor
    arm_actions_std: torch.Tensor
    arm_obs_mean: torch.Tensor
    arm_obs_std: torch.Tensor


class TaskMetrics(TypedDict):
    grasped: bool
    lifted: bool
    tray_reached: bool
    lowered_to_tray: bool
    released_over_tray: bool
    placement_success: bool


class TaskMetricsTracker:
    def __init__(
        self,
        *,
        controller: PickPlaceController,
        initial_cube_z: float,
    ) -> None:
        self.controller = controller
        self.initial_cube_z = initial_cube_z
        self.previous_cube_pos = controller.data.body("cube").xpos.copy()
        self.grasp_steps = 0
        self.lowering_steps = 0
        self.release_steps = 0
        self.placement_steps = 0
        self.grasped = False
        self.lifted = False
        self.tray_reached = False
        self.lowered_to_tray = False
        self.released_over_tray = False
        self.placement_success = False

    def update(self) -> None:
        cube_pos = self.controller.data.body("cube").xpos.copy()
        tray_pos = self.controller.data.site_xpos[self.controller.tray_center_id]
        tray_rot = self.controller.data.site_xmat[
            self.controller.tray_center_id
        ].reshape(3, 3)
        cube_tray_offset = tray_rot.T @ (cube_pos - tray_pos)
        two_finger_contact = self.controller.cube_has_two_finger_contact()
        any_finger_contact = self.controller.cube_has_any_finger_contact()

        self.grasp_steps = self.grasp_steps + 1 if two_finger_contact else 0
        if self.grasp_steps >= GRASP_STABLE_STEPS:
            self.grasped = True

        cube_lift = float(cube_pos[2] - self.initial_cube_z)
        if self.grasped and two_finger_contact and cube_lift >= CUBE_LIFT_MIN_DELTA:
            self.lifted = True

        tray_error = float(np.linalg.norm(cube_tray_offset[:2]))
        if self.lifted and tray_error <= TRAY_PLACE_TOL:
            self.tray_reached = True

        cube_inside_tray = bool(
            np.all(np.abs(cube_tray_offset[:2]) <= TRAY_INNER_XY_TOL)
        )
        cube_at_placement_height = (
            abs(float(cube_tray_offset[2])) <= PLACEMENT_Z_TOL
        )
        lowering_is_stable = (
            self.tray_reached
            and two_finger_contact
            and cube_inside_tray
            and cube_at_placement_height
        )
        self.lowering_steps = self.lowering_steps + 1 if lowering_is_stable else 0
        if self.lowering_steps >= LOWERING_STABLE_STEPS:
            self.lowered_to_tray = True

        release_is_stable = (
            self.lowered_to_tray
            and cube_inside_tray
            and cube_at_placement_height
            and not any_finger_contact
        )
        self.release_steps = self.release_steps + 1 if release_is_stable else 0
        if self.release_steps >= RELEASE_STABLE_STEPS:
            self.released_over_tray = True

        cube_speed = float(
            np.linalg.norm(cube_pos - self.previous_cube_pos) / self.controller.dt
        )
        self.previous_cube_pos = cube_pos
        placement_is_stable = (
            self.released_over_tray
            and cube_inside_tray
            and cube_at_placement_height
            and not any_finger_contact
            and cube_speed <= PLACEMENT_MAX_SPEED
        )
        self.placement_steps = self.placement_steps + 1 if placement_is_stable else 0
        if self.placement_steps >= PLACEMENT_STABLE_STEPS:
            self.placement_success = True

    def result(self) -> TaskMetrics:
        return {
            "grasped": self.grasped,
            "lifted": self.lifted,
            "tray_reached": self.tray_reached,
            "lowered_to_tray": self.lowered_to_tray,
            "released_over_tray": self.released_over_tray,
            "placement_success": self.placement_success,
        }


class EpisodeResult(TypedDict):
    steps: int
    seed: int | None
    phases: list[int]
    arm_disagreement: list[float]
    gripper_disagreement: list[bool]
    terminal_reason: str
    final_phase: int
    cube_init_pos: np.ndarray
    tray_init_pos: np.ndarray
    log_buffers: dict[str, list[np.ndarray]]
    observations: list[np.ndarray]
    expert_actions: list[np.ndarray]
    policy_actions: list[np.ndarray]
    executed_actions: list[np.ndarray]
    expert_action_mask: list[bool]
    task_metrics: TaskMetrics


def make_episode_seeds(*, seed: int, episodes: int) -> list[int]:
    seed_sequence = np.random.SeedSequence(seed)
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(episodes)
    ]


def select_dagger_control(
    *,
    mode: str,
    beta: float,
    arm_disagreement: float,
    gripper_disagreement: bool,
    intervention_threshold: float | None,
    intervention_steps: int,
    intervention_steps_remaining: int,
    rng: np.random.Generator,
) -> tuple[bool, int]:
    if mode == "beta":
        return rng.random() < beta, 0
    if mode != "threshold":
        raise ValueError(f"Unsupported DAgger intervention mode: {mode!r}")
    if intervention_threshold is None:
        raise ValueError("Threshold intervention mode requires a threshold")

    intervention_triggered = (
        arm_disagreement > intervention_threshold or gripper_disagreement
    )
    if intervention_triggered and intervention_steps_remaining == 0:
        intervention_steps_remaining = intervention_steps

    execute_expert = intervention_steps_remaining > 0
    if execute_expert:
        intervention_steps_remaining -= 1
    return execute_expert, intervention_steps_remaining


def initialize_dagger_worker(config: tuple[str, bool]) -> None:
    global _dagger_worker_state

    model_path, randomize_scene = config
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    model, normalize, action_space, norm_dict = load_policy(
        model_path=Path(model_path),
        device=device,
    )
    sim = SimEnv(randomize_scene=randomize_scene)
    _dagger_worker_state = (
        sim,
        model,
        device,
        normalize,
        action_space,
        norm_dict,
    )


def run_dagger_worker(
    task: tuple[int, int, int, float, str, str, float | None, int],
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
    ) = task
    sim, model, device, normalize, action_space, norm_dict = _dagger_worker_state
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
            dagger=True,
            beta=beta,
            intervention_mode=intervention_mode,
            intervention_threshold=intervention_threshold,
            intervention_steps=intervention_steps,
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


def load_policy(
    *,
    model_path: Path,
    device: torch.device,) -> tuple[MLP, bool, str, NormDict]:
    model = MLP().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
    intervention_mode: str,
    intervention_threshold: float | None,
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
    intervention_mode: str = "beta",
    intervention_threshold: float | None = None,
    intervention_steps: int = DEFAULT_INTERVENTION_STEPS,
    result: EpisodeResult,) -> None:
    dagger_path = dagger_dir / f"pick_place_{episode_idx:06d}.npz"
    np.savez_compressed(
        dagger_path,
        obs=np.asarray(result["observations"], dtype=np.float32),
        actions=np.asarray(result["expert_actions"], dtype=np.float32),
        policy_actions=np.asarray(result["policy_actions"], dtype=np.float32),
        executed_actions=np.asarray(result["executed_actions"], dtype=np.float32),
        execute_expert=np.asarray(result["expert_action_mask"], dtype=np.bool_),
        phases=np.asarray(result["phases"], dtype=np.int8),
        arm_disagreement=np.asarray(result["arm_disagreement"], dtype=np.float32),
        gripper_disagreement=np.asarray(result["gripper_disagreement"], dtype=np.bool_),
        terminal_reason=np.asarray(result["terminal_reason"]),
        final_phase=np.asarray(result["final_phase"], dtype=np.int8),
        seed=np.asarray(result["seed"], dtype=np.uint32),
        steps=np.asarray(result["steps"], dtype=np.int32),
        **{
            name: np.asarray(value, dtype=np.bool_)
            for name, value in result["task_metrics"].items()
        },
        beta=np.asarray(beta, dtype=np.float32),
        intervention_mode=np.asarray(intervention_mode),
        intervention_threshold=np.asarray(
            np.nan if intervention_threshold is None else intervention_threshold,
            dtype=np.float32,
        ),
        intervention_steps=np.asarray(intervention_steps, dtype=np.int32),
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
    intervention_mode: str = "beta",
    intervention_threshold: float | None = None,
    intervention_steps: int = DEFAULT_INTERVENTION_STEPS,
    rng: np.random.Generator,
    episode_seed: int | None = None,
    log_rollout: bool = False,
    viewer=None,
    should_stop=None,
    phase_callback=None,
    control_callback=None,) -> EpisodeResult:
    """Run one policy episode, stopping when the controller reaches DONE."""
    observations: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    policy_action_history: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    expert_action_mask: list[bool] = []
    phases: list[int] = []
    arm_disagreement: list[float] = []
    gripper_disagreement: list[bool] = []
    intervention_steps_remaining = 0

    sim.reset_episode()
    cube_init_pos = sim.data.qpos[sim.cube_qpos_addr : sim.cube_qpos_addr + 3].copy()
    tray_init_pos = sim.data.site("tray_center").xpos.copy()

    controller = (
        PickPlaceController(sim.model, sim.data) if track_phase or dagger else None
    )
    metrics_controller = controller or PickPlaceController(sim.model, sim.data)
    task_metrics = TaskMetricsTracker(
        controller=metrics_controller,
        initial_cube_z=float(cube_init_pos[2]),
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
            if controller is None:
                raise RuntimeError("DAgger rollout requires a controller")
            phases.append(controller.phase.value)

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
            current_arm_disagreement = float(
                np.linalg.norm(
                    expert_action[: ACTION_DIMS - 1]
                    - policy_action_np[: ACTION_DIMS - 1]
                )
            )
            arm_disagreement.append(current_arm_disagreement)
            current_gripper_disagreement = bool(
                (expert_action[ACTION_DIMS - 1] >= 127.5)
                != (policy_action_np[ACTION_DIMS - 1] >= 127.5)
            )
            gripper_disagreement.append(current_gripper_disagreement)
            (
                execute_expert_action,
                intervention_steps_remaining,
            ) = select_dagger_control(
                mode=intervention_mode,
                beta=beta,
                arm_disagreement=current_arm_disagreement,
                gripper_disagreement=current_gripper_disagreement,
                intervention_threshold=intervention_threshold,
                intervention_steps=intervention_steps,
                intervention_steps_remaining=intervention_steps_remaining,
                rng=rng,
            )

        action_to_execute = expert_action if execute_expert_action else policy_action_np
        if control_callback is not None:
            control_callback(steps, execute_expert_action)
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
        task_metrics.update()
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
        == len(phases)
        == len(arm_disagreement)
        == len(gripper_disagreement)
    ):
        raise ValueError("Dagger episode buffers have mismatched lengths")

    final_phase = controller.phase.value if controller is not None else -1
    terminal_reason = (
        "success" if controller is not None and controller.phase == Phase.DONE else "max_steps"
    )

    return {
        "steps": steps,
        "seed": episode_seed,
        "phases": phases,
        "arm_disagreement": arm_disagreement,
        "gripper_disagreement": gripper_disagreement,
        "terminal_reason": terminal_reason,
        "final_phase": final_phase,
        "cube_init_pos": cube_init_pos,
        "tray_init_pos": tray_init_pos,
        "log_buffers": log_buffers,
        "observations": observations,
        "expert_actions": expert_actions,
        "policy_actions": policy_action_history,
        "executed_actions": executed_actions,
        "expert_action_mask": expert_action_mask,
        "task_metrics": task_metrics.result(),
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
    intervention_mode: str = "beta",
    intervention_threshold: float | None = None,
    intervention_steps: int = DEFAULT_INTERVENTION_STEPS,
    create_dagger_subdir: bool = True,
    headless: bool,
    workers: int = 1,):
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
        intervention_mode=intervention_mode,
        intervention_threshold=intervention_threshold,
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

            episode_seeds = make_episode_seeds(seed=seed, episodes=episodes) if dagger else []
            for episode in episode_pbar:
                current_phase = None
                current_step = 0
                expert_was_executing = False
                last_expert_sound_at = float("-inf")
                episode_seed = episode_seeds[episode] if episode_seeds else None
                if episode_seed is not None:
                    sim.rng = np.random.default_rng(episode_seed)
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
                    intervention_mode=intervention_mode,
                    intervention_threshold=intervention_threshold,
                    intervention_steps=intervention_steps,
                    rng=np.random.default_rng(episode_seed) if episode_seed is not None else rng,
                    episode_seed=episode_seed,
                    log_rollout=log_rollout and log_dir is not None,
                    viewer=viewer,
                    should_stop=lambda: quit_requested,
                    phase_callback=update_phase_progress,
                    control_callback=update_control_progress,
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
        intervention_mode=args.dagger_mode,
        intervention_threshold=args.intervention_threshold,
        intervention_steps=args.intervention_steps,
        create_dagger_subdir=True,
        headless=args.headless,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
