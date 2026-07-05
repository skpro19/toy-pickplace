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

# def key_callback(keycode: int) -> None:
#     if chr(keycode).lower() == "q" and viewer_handle is not None:
#         viewer_handle.close()

EPSILON = 1e-6

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

            for steps in range(0,max_steps):
                # print(f"steps=>{steps}")
                obs = sim.build_observation()
                obs = torch.from_numpy(obs).to(device)
                
                obs_norm = torch.empty_like(obs)
                # normalize obs
                if normalize_obs: 
                    obs_norm = obs - arm_obs_mean
                    obs_norm = obs_norm / (arm_obs_std + EPSILON)
                else: 
                    obs_norm = obs
                    

                pred = model(obs_norm)
                actions = torch.empty_like(pred)

                if normalize_actions: 
                    actions[:7] = pred[:7] * (arm_actions_std + EPSILON) + arm_actions_mean
                    actions[7] = (255.0 if pred[7] >= 0.5 else 0)
                else:
                    actions = pred

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