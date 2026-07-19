"""Compare free-camera and top-camera renders; save side-by-side PNGs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "panda_pick_place.xml"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
DEFAULT_OUT_DIR = PLOTS_DIR / "top-camera-calibration"
DEFAULT_CAPTURE_DIR = PLOTS_DIR / "captured_camera_pose"

FREE_CAMERA_LOOKAT = (0.45, 0.0, 0.78)
FREE_CAMERA_DISTANCE = 1.35
FREE_CAMERA_AZIMUTH = 145.0
FREE_CAMERA_ELEVATION = -25.0
FIXED_CAMERA_NAME = "top-camera"
DEFAULT_CAMERA_FOVY = 45.0
DEFAULT_CAPTURE_RENDER_SIZE = 256


def make_capture_dir(*, capture_root: Path = DEFAULT_CAPTURE_DIR) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return capture_root / stamp


def make_capture_paths(*, capture_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        capture_dir / "camera_pose.txt",
        capture_dir / "free_camera.png",
        capture_dir / "top_camera.png",
        capture_dir / "side_by_side.png",
    )


def find_reset_key(*, model: mujoco.MjModel) -> int:
    for name in ("task_home", "home"):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id >= 0:
            return key_id
    return -1


def reset_home(*, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = find_reset_key(model=model)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        if model.nu:
            data.ctrl[: model.nu] = model.key_ctrl[key_id, : model.nu]
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def make_free_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = FREE_CAMERA_LOOKAT
    cam.distance = FREE_CAMERA_DISTANCE
    cam.azimuth = FREE_CAMERA_AZIMUTH
    cam.elevation = FREE_CAMERA_ELEVATION
    return cam


def camera_pose_from_mjv_camera(
    *,
    data: mujoco.MjData,
    cam: mujoco.MjvCamera,
) -> dict[str, np.ndarray | float]:
    """Convert a free-camera pose to fixed-camera XML fields."""
    mujoco.mj_forward(data.model, data)
    headpos = np.zeros(3)
    forward = np.zeros(3)
    up = np.zeros(3)
    right = np.zeros(3)
    mujoco.mjv_cameraFrame(headpos, forward, up, right, data, cam)
    return {
        "lookat": cam.lookat.copy(),
        "distance": float(cam.distance),
        "azimuth": float(cam.azimuth),
        "elevation": float(cam.elevation),
        "pos": headpos,
        "xyaxes": np.concatenate([right, up]),
    }


def format_camera_pose_text(
    *,
    pose: dict[str, np.ndarray | float],
    camera_name: str = FIXED_CAMERA_NAME,
    fovy: float = DEFAULT_CAMERA_FOVY,
) -> str:
    lookat = pose["lookat"]
    pos = pose["pos"]
    xyaxes = pose["xyaxes"]
    return "\n".join(
        [
            "# free-camera params (for verify_top_camera.py constants)",
            f"FREE_CAMERA_LOOKAT = ({lookat[0]:.6f}, {lookat[1]:.6f}, {lookat[2]:.6f})",
            f"FREE_CAMERA_DISTANCE = {pose['distance']:.6f}",
            f"FREE_CAMERA_AZIMUTH = {pose['azimuth']:.6f}",
            f"FREE_CAMERA_ELEVATION = {pose['elevation']:.6f}",
            "",
            f"# paste into scenes/panda_pick_place.xml as <camera name=\"{camera_name}\">",
            f'<camera name="{camera_name}"',
            f'  pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"',
            f'  xyaxes="{xyaxes[0]:.6f} {xyaxes[1]:.6f} {xyaxes[2]:.6f} '
            f'{xyaxes[3]:.6f} {xyaxes[4]:.6f} {xyaxes[5]:.6f}"',
            f'  fovy="{fovy:.6g}"/>',
        ]
    )


def save_captured_camera_pose(
    *,
    pose: dict[str, np.ndarray | float],
    out_path: Path,
    camera_name: str = FIXED_CAMERA_NAME,
    fovy: float = DEFAULT_CAMERA_FOVY,
) -> Path:
    text = format_camera_pose_text(pose=pose, camera_name=camera_name, fovy=fovy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    return out_path


def capture_camera_pose(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam: mujoco.MjvCamera,
    capture_dir: Path | None = None,
    camera_name: str = FIXED_CAMERA_NAME,
    fovy: float = DEFAULT_CAMERA_FOVY,
    render_size: int = DEFAULT_CAPTURE_RENDER_SIZE,
) -> Path:
    if capture_dir is None:
        capture_dir = make_capture_dir()
    pose_path, free_image_path, top_image_path, side_by_side_path = make_capture_paths(
        capture_dir=capture_dir,
    )

    pose = camera_pose_from_mjv_camera(data=data, cam=cam)
    saved_pose_path = save_captured_camera_pose(
        pose=pose,
        out_path=pose_path,
        camera_name=camera_name,
        fovy=fovy,
    )

    with mujoco.Renderer(model, render_size, render_size) as renderer:
        img_free, img_top = render_pair(
            model=model,
            data=data,
            renderer=renderer,
            free_cam=cam,
            fixed_camera_name=camera_name,
        )
    save_single(image=img_free, out_path=free_image_path)
    save_single(image=img_top, out_path=top_image_path)
    save_side_by_side(
        img_free=img_free,
        img_fixed=img_top,
        out_path=side_by_side_path,
        title=capture_dir.name,
    )

    print(format_camera_pose_text(pose=pose, camera_name=camera_name, fovy=fovy))
    print(f"\nSaved capture artifacts under: {capture_dir.resolve()}")
    print(f"  {saved_pose_path.name}")
    print(f"  {free_image_path.name}")
    print(f"  {top_image_path.name}")
    print(f"  {side_by_side_path.name}")
    return capture_dir


def render_pair(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    free_cam: mujoco.MjvCamera | None = None,
    fixed_camera_name: str = FIXED_CAMERA_NAME,
) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_forward(model, data)
    if free_cam is None:
        free_cam = make_free_camera()

    renderer.update_scene(data, camera=free_cam)
    img_free = renderer.render().copy()

    renderer.update_scene(data, camera=fixed_camera_name)
    img_fixed = renderer.render().copy()
    return img_free, img_fixed


def pixel_diff_stats(*, img_a: np.ndarray, img_b: np.ndarray) -> tuple[int, float]:
    diff = np.abs(img_a.astype(np.int16) - img_b.astype(np.int16))
    return int(diff.max()), float(diff.mean())


def save_side_by_side(
    *,
    img_free: np.ndarray,
    img_fixed: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_free)
    axes[0].set_title("free camera (viewer default)")
    axes[0].axis("off")

    axes[1].imshow(img_fixed)
    axes[1].set_title(FIXED_CAMERA_NAME)
    axes[1].axis("off")

    max_diff, mean_diff = pixel_diff_stats(img_a=img_free, img_b=img_fixed)
    fig.suptitle(f"{title} | max diff={max_diff}, mean diff={mean_diff:.2f}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_single(*, image: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, image)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render free vs top-camera views and save comparison PNGs."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for PNG outputs.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, FIXED_CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"Camera '{FIXED_CAMERA_NAME}' not found in {SCENE_PATH}")

    out_dir = args.out_dir
    with mujoco.Renderer(model, args.height, args.width) as renderer:
        reset_home(model=model, data=data)
        img_free, img_fixed = render_pair(model=model, data=data, renderer=renderer)
        save_single(image=img_free, out_path=out_dir / "home_free_camera.png")
        save_single(image=img_fixed, out_path=out_dir / "home_top_camera.png")
        save_side_by_side(
            img_free=img_free,
            img_fixed=img_fixed,
            out_path=out_dir / "home_side_by_side.png",
            title="task_home",
        )
        max_diff, mean_diff = pixel_diff_stats(img_a=img_free, img_b=img_fixed)
        print(f"{FIXED_CAMERA_NAME} id={cam_id}")
        print(f"task_home pixel diff: max={max_diff}, mean={mean_diff:.4f}")
        print(f"Saved PNGs under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
