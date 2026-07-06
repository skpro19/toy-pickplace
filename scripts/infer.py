import torch 
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import PickPlaceDataset
from train import MLP
from sim import SimEnv
from data import DataCollector
from formatters import print_1d_array, print_1d_tensor
import mujoco
import mujoco.viewer
import time
import numpy as np

from constant import (
        EPSILON, 
        ACTION_DIMS
        )

def infer(*, model_path: str):
    
   
    sim = SimEnv()
    sim.reset_episode()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path(model_path)
    model = MLP().to(device)
    
    ckpt = torch.load(model_path, weights_only=False)

    model.load_state_dict(ckpt["model_dict"])
    
    
    normalize_actions: bool = ckpt["normalize_actions"]
    arm_actions_mean: np.ndarray = ckpt["arm_actions_mean"]
    arm_actions_std: np.ndarray = ckpt["arm_actions_std"]

    normalize_obs: bool = ckpt["normalize_obs"]
    arm_obs_mean: np.ndarray = ckpt["arm_obs_mean"]
    arm_obs_std: np.ndarray = ckpt["arm_obs_std"]

    # debug
    assert normalize_actions, "normalize_actions must be True"
    assert normalize_obs, "normalize_obs must be True"

    if normalize_actions:
        arm_actions_mean: torch.Tensor = torch.from_numpy(arm_actions_mean).to(device)
        arm_actions_std: torch.Tensor = torch.from_numpy(arm_actions_std).to(device)

        arm_actions_mean = arm_actions_mean.squeeze(0)
        arm_actions_std = arm_actions_std.squeeze(0)

    if normalize_obs:
        arm_obs_mean: torch.Tensor = torch.from_numpy(arm_obs_mean).to(device)
        arm_obs_std: torch.Tensor = torch.from_numpy(arm_obs_std).to(device)

        arm_obs_mean = arm_obs_mean.squeeze(0)
        arm_obs_std = arm_obs_std.squeeze(0)

    model.eval() 
    

    max_steps = 100 * 100 * 100
    # max_steps = 1

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        # key_callback=key_callback,
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

            for _ in range(0,max_steps):
                # print(f"steps=>{steps}")
                obs = sim.build_observation()
                obs = torch.from_numpy(obs).to(device).unsqueeze(0)
                
                obs_norm = torch.empty_like(obs)
                # normalize obs
                if normalize_obs: 
                    obs_norm = obs - arm_obs_mean
                    obs_norm = obs_norm / (arm_obs_std + EPSILON)
                else: 
                    obs_norm = obs

                # print(f"type(obs_norm)=>{type(obs_norm)}")
                # print(f"obs_norm.shape=>{obs_norm.shape}")
                    

                # pred = model(obs_norm)
                actions = torch.empty(obs_norm.shape[0], ACTION_DIMS)

                (joints_pred, gripper_pred) = model(obs_norm)
                
                # print(f"joints_pred.shape=>{joints_pred.shape}")
                # print(f"gripper_pred.shape=>{gripper_pred.shape}")
                
                joints_actions = torch.empty_like(joints_pred)
                gripper_actions = torch.empty_like(gripper_pred)


                # unnormalize actions
                if normalize_actions: 
                    joints_actions = joints_pred * (arm_actions_std + EPSILON) + arm_actions_mean
                    # gripper_actions = (255.0 if gripper_pred >= 0.5 else 0)
                    gripper_actions = torch.where(torch.sigmoid(gripper_pred) >= 0.5, 255.0, 0.0)

                # print(f"type(joints_actions)=>{type(joints_actions)}")
                # print(f"type(gripper_actions)=>{type(gripper_actions)}")
                actions = torch.concat([joints_actions, gripper_actions], dim=1)
                

                sim.data.ctrl[:sim.model.nu] = actions.detach().cpu().numpy()
                
                mujoco.mj_step(sim.model, sim.data)
                time.sleep(sim.model.opt.timestep * 10)

                viewer.sync()

                # break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model path e.g. checkpoints/026_mlp_action_norm")
    return parser.parse_args()

def main(): 
    args = parse_args()
    infer(model_path=args.model)


if __name__ == "__main__":
    main()