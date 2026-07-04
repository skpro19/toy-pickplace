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

# def key_callback(keycode: int) -> None:
#     if chr(keycode).lower() == "q" and viewer_handle is not None:
#         viewer_handle.close()

def infer(*, model_path: str):
    
   
    sim = SimEnv()
    sim.reset_episode()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path(model_path)
    model = MLP().to(device)
    
    ckpt = torch.load(model_path, weights_only=False)

    model.load_state_dict(ckpt["model_dict"])
    
    NORMALIZE_ACTIONS = ckpt["normalize_actions"]
    ARM_ACTIONS_MEAN = ckpt["arm_actions_mean"]
    ARM_ACTIONS_STD = ckpt["arm_actions_std"]

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

                # print(f"type(obs)=>{type(obs)} obs.shape=>{obs.shape} obs.dtype=>{obs.dtype}")

                obs_tensor = torch.from_numpy(obs)
                obs_tensor = obs_tensor.to(device)

                # print(f"type(obs_tensor)=>{type(obs_tensor)} obs_tensor.shape=>{obs_tensor.shape} obs_tensor.dtype=>{obs_tensor.dtype}")

                pred = model(obs_tensor)

                if NORMALIZE_ACTIONS: 
                    # torch.from
                    ARM_ACTIONS_STD_tensor = torch.from_numpy(ARM_ACTIONS_STD).to(device)
                    ARM_ACTIONS_MEAN_tensor = torch.from_numpy(ARM_ACTIONS_MEAN).to(device)

                    print(f"type(ARM_ACTIONS_STD_tensor)=>{type(ARM_ACTIONS_STD_tensor)} ARM_ACTIONS_STD_tensor.shape=>{ARM_ACTIONS_STD_tensor.shape} ARM_ACTIONS_STD_tensor.device=>{ARM_ACTIONS_STD_tensor.device}")
                    print(f"type(ARM_ACTIONS_MEAN_tensor)=>{type(ARM_ACTIONS_MEAN_tensor)} ARM_ACTIONS_MEAN_tensor.shape=>{ARM_ACTIONS_MEAN_tensor.shape} ARM_ACTIONS_MEAN_tensor.device=>{ARM_ACTIONS_MEAN_tensor.device}")
                    

                    pred[:7] = pred[:7] * (ARM_ACTIONS_STD_tensor + 1e-6) + ARM_ACTIONS_MEAN_tensor
                    pred[7] = (255.0 if pred[7] >= 0.5 else 0)
                    print(f"type(pred)=>{type(pred)} pred.shape=>{pred.shape} pred.device=>{pred.device}")
                    # print(f"pred[7]=>{pred[7]}")
                    print(f"type(ARM_ACTIONS_MEAN)=>{type(ARM_ACTIONS_MEAN)} ARM_ACTIONS_MEAN.shape=>{ARM_ACTIONS_MEAN.shape}")
                # print(f"type(pred)=>{type(pred)} pred.shape=>{pred.shape} pred.dtype=>{pred.dtype}")
                # print_1d_tensor(tensor=pred, label="pred")

                # update mujoco data
                # print_1d_array(array=sim.data.ctrl, length=sim.model.nu, label="[Before update] sim.data.ctrl")
                sim.data.ctrl[:sim.model.nu] = pred.detach().cpu().numpy()
                # print_1d_array(array=sim.data.ctrl, length=sim.model.nu, label="[after update] sim.data.ctrl")

                mujoco.mj_step(sim.model, sim.data)
                time.sleep(sim.model.opt.timestep * 10)

                viewer.sync()

                # break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default ="checkpoints/test/model.pt")
    return parser.parse_args()

def main(): 
    args = parse_args()
    infer(model_path=args.model)


if __name__ == "__main__":
    main()