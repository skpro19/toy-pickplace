"""Policy inference and single-episode rollout runtime."""

import time
from typing import TypedDict

import mujoco
import numpy as np

from constant import ACTION_DIMS
from expert import Phase, PickPlaceController
from policy_runtime.types import PolicyRuntime
from rollout_core.dagger import DEFAULT_INTERVENTION_STEPS, select_dagger_control
from rollout_core.metrics import TaskMetrics, TaskMetricsTracker
from rollout_core.persistence import append_step_log
from sim import SimEnv


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


def run_policy_episode(
    *,
    sim: SimEnv,
    runtime: PolicyRuntime,
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
    control_callback=None,
) -> EpisodeResult:
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
    cube_init_pos = sim.data.qpos[
        sim.cube_qpos_addr : sim.cube_qpos_addr + 3
    ].copy()
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
        obs = runtime.observe(sim=sim)
        if dagger:
            observations.append(obs.copy())
            if controller is None:
                raise RuntimeError("DAgger rollout requires a controller")
            phases.append(controller.phase.value)

        step = runtime.act(sim=sim, observation=obs)
        policy_action_np = step["action"]

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
                obs=step["obs_tensor"],
                actions=step["policy_actions"],
                executed_actions=action_to_execute,
                joints_pred_raw=step["joints_pred"],
                joints_pred_unnorm=step["joints_pred_unnorm"],
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
        "success"
        if controller is not None and controller.phase == Phase.DONE
        else "max_steps"
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
