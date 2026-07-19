from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from constant import CAMERA_NAME, IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402
from sim import SimEnv  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "smoke_test_render"
BENCHMARK_FRAMES = 300
MIN_COLOR_PIXELS = 3


def count_red_pixels(*, image: np.ndarray) -> int:
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    return int(np.sum((red >= 100) & (green <= 90) & (blue <= 90)))


def count_blue_pixels(*, image: np.ndarray) -> int:
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    return int(np.sum((blue >= 100) & (red <= 90) & (green <= 110)))


def assert_image_valid(*, image: np.ndarray, label: str) -> None:
    expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    assert image.shape == expected_shape, (
        f"{label}: expected shape {expected_shape}, got {image.shape}"
    )
    assert image.dtype == np.uint8, f"{label}: expected uint8, got {image.dtype}"

    red_pixels = count_red_pixels(image=image)
    blue_pixels = count_blue_pixels(image=image)
    assert red_pixels >= MIN_COLOR_PIXELS, (
        f"{label}: expected red cube pixels (>={MIN_COLOR_PIXELS}), got {red_pixels}"
    )
    assert blue_pixels >= MIN_COLOR_PIXELS, (
        f"{label}: expected blue tray pixels (>={MIN_COLOR_PIXELS}), got {blue_pixels}"
    )
    print(
        f"{label}: shape={image.shape}, dtype={image.dtype}, "
        f"red_pixels={red_pixels}, blue_pixels={blue_pixels}"
    )


def save_image(*, image: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, image)


def benchmark_render_fps(*, sim: SimEnv, frames: int) -> float:
    start = time.perf_counter()
    for _ in range(frames):
        sim.build_image()
    elapsed = time.perf_counter() - start
    return frames / elapsed


def main() -> None:
    cam_id_check = SimEnv()
    try:
        import mujoco

        cam_id = mujoco.mj_name2id(
            cam_id_check.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            CAMERA_NAME,
        )
        assert cam_id >= 0, f"Camera '{CAMERA_NAME}' not found in scene"
    finally:
        cam_id_check.close()

    sim_home = SimEnv()
    try:
        sim_home.reset_home()
        home_image = sim_home.build_image()
        assert_image_valid(image=home_image, label="task_home")
        save_image(image=home_image, out_path=OUT_DIR / "home_top_camera.png")

        fps = benchmark_render_fps(sim=sim_home, frames=BENCHMARK_FRAMES)
        ms_per_frame = 1000.0 / fps
        print(
            f"render fps ({IMAGE_WIDTH}x{IMAGE_HEIGHT}, n={BENCHMARK_FRAMES}): "
            f"{fps:.1f} fps ({ms_per_frame:.2f} ms/frame)"
        )
    finally:
        sim_home.close()

    sim_rand = SimEnv(randomize_scene=True, seed=0)
    try:
        sim_rand.reset_episode()
        rand_image = sim_rand.build_image()
        assert_image_valid(image=rand_image, label="randomized_seed_0")
        save_image(image=rand_image, out_path=OUT_DIR / "randomized_seed_0_top_camera.png")
    finally:
        sim_rand.close()

    print(f"Saved PNGs under {OUT_DIR.resolve()}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
