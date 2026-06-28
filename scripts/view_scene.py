from pathlib import Path
import time

import numpy as np
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


def add_site_markers(model: mujoco.MjModel, data: mujoco.MjData, viewer: mujoco.viewer.Handle) -> None:
    viewer.user_scn.ngeom = 0
    for index, site_name in enumerate(("cube_center", "cube_hover", "tray_center", "tray_hover")):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0 or index >= viewer.user_scn.maxgeom:
            continue

        rgba = model.site_rgba[site_id].copy()
        rgba[3] = 1.0
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[index],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.01, 0.0, 0.0]),
            pos=data.site_xpos[site_id],
            mat=np.eye(3).ravel(),
            rgba=rgba,
        )
        viewer.user_scn.geoms[index].label = site_name
        viewer.user_scn.ngeom = index + 1


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    reset_home(model, data)
    home_id = find_reset_key(model)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (0.45, 0.0, 0.78)
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25

        while viewer.is_running():
            if model.nu and home_id >= 0:
                data.ctrl[: model.nu] = model.key_ctrl[home_id, : model.nu]
            mujoco.mj_step(model, data)
            add_site_markers(model, data, viewer)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
