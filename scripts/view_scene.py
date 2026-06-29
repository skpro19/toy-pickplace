from pathlib import Path
import time

import mujoco
import mujoco.viewer


SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "panda_pick_place.xml"


def find_reset_key(model: mujoco.MjModel) -> int:
    for name in ("task_home", "home"):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id >= 0:
            return key_id
    return -1


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = find_reset_key(model)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        if model.nu:
            data.ctrl[: model.nu] = model.key_ctrl[key_id, : model.nu]
    else:
        mujoco.mj_resetData(model, data)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    reset_home(model, data)
    home_id = find_reset_key(model)
    viewer_handle: mujoco.viewer.Handle | None = None

    def key_callback(keycode: int) -> None:
        if chr(keycode).lower() == "q" and viewer_handle is not None:
            viewer_handle.close()

    with mujoco.viewer.launch_passive(
        model,
        data,
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

        while viewer.is_running():
            if model.nu and home_id >= 0:
                data.ctrl[: model.nu] = model.key_ctrl[home_id, : model.nu]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
