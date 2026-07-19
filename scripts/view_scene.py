from pathlib import Path
import time

import mujoco
import mujoco.viewer

from verify_top_camera import (
    DEFAULT_CAMERA_FOVY,
    DEFAULT_CAPTURE_DIR,
    FIXED_CAMERA_NAME,
    capture_camera_pose,
    make_free_camera,
)


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


def fixed_camera_fovy(*, model: mujoco.MjModel) -> float:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, FIXED_CAMERA_NAME)
    if cam_id < 0:
        return DEFAULT_CAMERA_FOVY
    return float(model.cam_fovy[cam_id])


def apply_free_camera(*, viewer_cam: mujoco.MjvCamera) -> None:
    free_cam = make_free_camera()
    viewer_cam.type = free_cam.type
    viewer_cam.lookat[:] = free_cam.lookat
    viewer_cam.distance = free_cam.distance
    viewer_cam.azimuth = free_cam.azimuth
    viewer_cam.elevation = free_cam.elevation


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    reset_home(model, data)
    home_id = find_reset_key(model)
    viewer_handle: mujoco.viewer.Handle | None = None
    camera_fovy = fixed_camera_fovy(model=model)

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower()
        if key == "q" and viewer_handle is not None:
            viewer_handle.close()
        if key == "p" and viewer_handle is not None:
            capture_camera_pose(
                model=model,
                data=data,
                cam=viewer_handle.cam,
                camera_name=FIXED_CAMERA_NAME,
                fovy=camera_fovy,
            )

    print("Controls: orbit/zoom/pan with mouse, press 'p' to capture camera pose, 'q' to quit.")
    print(
        "Captured artifacts are saved under "
        f"{DEFAULT_CAPTURE_DIR.resolve()}/<YYYY-MM-DD_HH-MM-SS>/"
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        viewer_handle = viewer
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        apply_free_camera(viewer_cam=viewer.cam)

        while viewer.is_running():
            if model.nu and home_id >= 0:
                data.ctrl[: model.nu] = model.key_ctrl[home_id, : model.nu]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
