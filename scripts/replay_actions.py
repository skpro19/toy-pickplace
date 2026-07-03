""" Load and replay actions from a .npz file in mujoco viewer."""
import numpy as np 
import mujoco, mujoco.viewer
import time

from sim import SimEnv

def replay_actions(*, file:str) -> None: 
    
    sim = SimEnv()

    with np.load(file) as data:
        obs = data["obs"]
        actions = data["actions"]

        print(f"actions.shape=>{actions.shape}")
        # print(f"actions=>{actions}")

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

            for idx, action in enumerate(actions): 
                # print(f"action.shape=>{action.shape}")
                sim.data.ctrl[:sim.model.nu] = action
                mujoco.mj_step(sim.model, sim.data)
                # time.sleep(sim.model.opt.timestep * 1)
                viewer.sync()
                



def main():
    file = "data/demos/test/test.npz"
    replay_actions(file=file)


if __name__ == "__main__":
    main()
