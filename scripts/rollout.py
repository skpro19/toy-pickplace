import torch 
import argparse
import json
from pathlib import Path
from datetime import datetime

from mlp import MLP
from sim import SimEnv
import mujoco
import mujoco.viewer
import time
import numpy as np

from constant import (
        EPSILON, 
        ACTION_DIMS,
        MAX_ARM_DELTA,
        )


def make_rollout_log_dir(*, log_root: Path, model_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_name = model_path.parent.name or model_path.stem
    return log_root / f"{timestamp}_{checkpoint_name}"


def append_step_log(
    *,
    buffers: dict[str, list[np.ndarray]],
    obs: torch.Tensor,
    actions: torch.Tensor,
    joints_pred_raw: torch.Tensor,
    joints_pred_unnorm: torch.Tensor,
) -> None:
    buffers["rollout_obs"].append(obs.squeeze(0).detach().cpu().numpy())
    buffers["rollout_actions"].append(actions.squeeze(0).detach().cpu().numpy())
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
    buffers: dict[str, list[np.ndarray]],
) -> None:
    train_episode_path = train_npz_dir / f"pick_place_{episode_idx:06d}.npz"
    if not train_episode_path.exists():
        raise FileNotFoundError(f"Training episode file not found: {train_episode_path}")

    with np.load(train_episode_path) as data:
        missing_keys = {"obs", "actions"} - set(data.files)
        if missing_keys:
            raise KeyError(f"{train_episode_path} is missing keys: {sorted(missing_keys)}")
        train_obs = np.asarray(data["obs"], dtype=np.float32)
        train_actions = np.asarray(data["actions"], dtype=np.float32)

    rollout_obs = np.asarray(buffers["rollout_obs"], dtype=np.float32)
    rollout_actions = np.asarray(buffers["rollout_actions"], dtype=np.float32)
    joints_pred_raw = np.asarray(buffers["joints_pred_raw"], dtype=np.float32)
    joints_pred_unnorm = np.asarray(buffers["joints_pred_unnorm"], dtype=np.float32)

    arrays = {
        "rollout_obs": rollout_obs,
        "train_obs": train_obs,
        "rollout_actions": rollout_actions,
        "train_actions": train_actions,
        "joints_pred_raw": joints_pred_raw,
        "joints_pred_unnorm": joints_pred_unnorm,
    }
    if action_space == "joint_delta":
        arrays["rollout_action_deltas"] = rollout_actions[:, : ACTION_DIMS - 1] - rollout_obs[:, : ACTION_DIMS - 1]
        arrays["train_action_deltas"] = train_actions[:, : ACTION_DIMS - 1] - train_obs[:, : ACTION_DIMS - 1]

    episode_path = log_dir / f"episode_{episode_idx:06d}.npz"
    np.savez_compressed(episode_path, **arrays)


def rollout(
    *,
    model_path: str,
    randomize_scene: bool,
    seed: int,
    episodes: int,
    max_steps: int,
    log_root: str | Path,
    train_npz_dir: str | Path,
    log_rollout: bool,
):
    
   
    # sim = SimEnv()
    sim = SimEnv(randomize_scene=randomize_scene, seed=seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path(model_path)
    model = MLP().to(device)
    
    ckpt = torch.load(model_path, weights_only=False)

    model.load_state_dict(ckpt["model_dict"])
    
    
    normalize = ckpt["normalize"]
    action_space = ckpt["action_space"]
    arm_actions_mean = ckpt["arm_actions_mean"]
    arm_actions_std = ckpt["arm_actions_std"]
    arm_obs_mean = ckpt["arm_obs_mean"]
    arm_obs_std = ckpt["arm_obs_std"]



    # normalize_actions: bool = ckpt["normalize_actions"]
    # arm_actions_mean: np.ndarray = ckpt["arm_actions_mean"]
    # arm_actions_std: np.ndarray = ckpt["arm_actions_std"]

    # normalize_obs: bool = ckpt["normalize_obs"]
    # arm_obs_mean: np.ndarray = ckpt["arm_obs_mean"]
    # arm_obs_std: np.ndarray = ckpt["arm_obs_std"]

    # debug
    assert normalize, "normalize must be True"
    assert action_space in ["joint_delta", "absolute"], "action_space must be either joint_delta or absolute"

    log_dir = None
    train_npz_dir = Path(train_npz_dir)
    if log_rollout:
        if not train_npz_dir.exists():
            raise FileNotFoundError(f"Training npz directory not found: {train_npz_dir}")
        log_dir = make_rollout_log_dir(log_root=Path(log_root), model_path=model_path)
        log_dir.mkdir(parents=True, exist_ok=False)
        # metadata = {
        #     "model_path": str(model_path),
        #     "train_npz_dir": str(train_npz_dir),
        #     "action_space": action_space,
        #     "normalize": normalize,
        #     "seed": seed,
        #     "randomize_scene": randomize_scene,
        #     "episodes": episodes,
        #     "max_steps": max_steps,
        #     "device": str(device),
        #     "logged_arrays": [
        #         "rollout_obs",
        #         "train_obs",
        #         "rollout_actions",
        #         "train_actions",
        #     ],
        #     "episode_alignment": "rollout episode i uses train pick_place_i.npz",
        # }
        # if action_space == "joint_delta":
        #     metadata["logged_arrays"].extend(
        #         ["rollout_action_deltas", "train_action_deltas"]
        #     )
        # with (log_dir / "metadata.json").open("w", encoding="utf-8") as f:
        #     json.dump(metadata, f, indent=2)
        print(f"Rollout log dir: {log_dir}")

    if normalize:
        arm_actions_mean: torch.Tensor = torch.from_numpy(arm_actions_mean).to(device).squeeze(0)
        arm_actions_std: torch.Tensor = torch.from_numpy(arm_actions_std).to(device).squeeze(0)

        arm_obs_mean: torch.Tensor = torch.from_numpy(arm_obs_mean).to(device).squeeze(0)
        arm_obs_std: torch.Tensor = torch.from_numpy(arm_obs_std).to(device).squeeze(0)

    model.eval() 
    

    quit_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal quit_requested
        if chr(keycode).lower() == "q":
            quit_requested = True

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        viewer_handle = viewer
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (0.45, 0.0, 0.78)
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25
    
        with torch.no_grad():  

            for episode in range(episodes):
                steps = 0
                sim.reset_episode()
                log_buffers: dict[str, list[np.ndarray]] = {
                    "rollout_obs": [],
                    "rollout_actions": [],
                    "joints_pred_raw": [],
                    "joints_pred_unnorm": [],
                }

                while viewer.is_running() and not quit_requested and steps < max_steps:
                    # print(f"steps=>{steps}")
                    obs = sim.build_observation()
                    obs = torch.from_numpy(obs).to(device).unsqueeze(0)
                    
                    obs_norm = torch.empty_like(obs)
                    # normalize obs
                    if normalize: 
                        obs_norm = obs - arm_obs_mean
                        obs_norm = obs_norm / (arm_obs_std + EPSILON)
                    else: 
                        obs_norm = obs

                    # print(f"type(obs_norm)=>{type(obs_norm)}")
                    # print(f"obs_norm.shape=>{obs_norm.shape}")
                        

                    # pred = model(obs_norm)
                    (joints_pred, gripper_pred) = model(obs_norm)
                    
                    # print(f"joints_pred.shape=>{joints_pred.shape}")
                    # print(f"gripper_pred.shape=>{gripper_pred.shape}")
                    
                    # unnormalize actions
                    if normalize: 
                        joints_pred_unnorm = joints_pred * (arm_actions_std + EPSILON) + arm_actions_mean
                        # gripper_actions = (255.0 if gripper_pred >= 0.5 else 0)
                    else:
                        joints_pred_unnorm = joints_pred

                    joints_actions = joints_pred_unnorm

                    arm_qpos = obs[:, 0:ACTION_DIMS-1]
                      
                    if action_space == "joint_delta":
                        joints_actions = torch.clamp(
                            joints_actions,
                            min=-MAX_ARM_DELTA,
                            max=MAX_ARM_DELTA,
                        )
                        joints_actions = joints_actions + arm_qpos
                    gripper_prob = torch.sigmoid(gripper_pred)
                    gripper_actions = torch.where(gripper_prob >= 0.5, 255.0, 0.0)

                    # print(f"type(joints_actions)=>{type(joints_actions)}")
                    # print(f"type(gripper_actions)=>{type(gripper_actions)}")
                    actions = torch.concat([joints_actions, gripper_actions], dim=1)

                    if log_rollout and log_dir is not None:
                        append_step_log(
                            buffers=log_buffers,
                            obs=obs,
                            actions=actions,
                            joints_pred_raw=joints_pred,
                            joints_pred_unnorm=joints_pred_unnorm,
                        )
                     

                    sim.data.ctrl[:sim.model.nu] = actions.squeeze(0).detach().cpu().numpy()
                    
                    mujoco.mj_step(sim.model, sim.data)
                    time.sleep(sim.model.opt.timestep * 10)

                    viewer.sync()
                    steps += 1

                if log_rollout and log_dir is not None:
                    save_episode_log(
                        log_dir=log_dir,
                        train_npz_dir=train_npz_dir,
                        episode_idx=episode,
                        action_space=action_space,
                        buffers=log_buffers,
                    )

                # break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model path e.g. checkpoints/026_mlp_action_norm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--randomize-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs/rollouts"))
    parser.add_argument("--train-npz-dir", type=Path, default=Path("data/rand-100"))
    parser.add_argument(
        "--log-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()

def main(): 
    args = parse_args()
    rollout(model_path=args.model, 
        randomize_scene=args.randomize_scene, 
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        log_root=args.log_dir,
        train_npz_dir=args.train_npz_dir,
        log_rollout=args.log_rollout)


if __name__ == "__main__":
    main()
