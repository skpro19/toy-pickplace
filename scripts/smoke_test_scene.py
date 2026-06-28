from pathlib import Path

import mujoco


SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "panda_pick_place.xml"


def find_reset_key(model: mujoco.MjModel) -> int:
    for name in ("task_home", "home"):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id >= 0:
            return key_id
    return -1


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    key_id = find_reset_key(model)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        if model.nu:
            data.ctrl[: model.nu] = model.key_ctrl[key_id, : model.nu]

    for _ in range(200):
        mujoco.mj_step(model, data)

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    tray_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
    cube_pos = data.xpos[cube_id]
    tray_pos = data.xpos[tray_id]

    print(f"Loaded scene: {SCENE_PATH}")
    print(f"Bodies: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"Cube world position: {cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}")
    print(f"Tray world position: {tray_pos[0]:.3f}, {tray_pos[1]:.3f}, {tray_pos[2]:.3f}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
