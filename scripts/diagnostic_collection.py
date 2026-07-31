"""Cross-host reproducibility diagnostic: collect expert episodes with full float64 state.

Saves per-frame MuJoCo state (qpos, qvel, ctrl, body positions, body matrices)
alongside standard observations, actions, and images to enable frame-level
cross-host comparison.

Usage:
    CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        uv run python scripts/diagnostic_collection.py \
        --seed 0 --episodes 100 --capture-hz 60 --out-dir data/diagnostic/seed0-hostA
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constant import DEFAULT_CAPTURE_HZ
from expert import Phase, PickPlaceController
from reproducibility import (
    _command_output,
    _cpu_model,
    _opengl_info,
    _package_versions,
    configure_reproducibility,
)
from sim import SimEnv


def _gpu_uuid_vbios() -> tuple[str | None, str | None]:
    output = _command_output(
        command=[
            "nvidia-smi",
            "--query-gpu=uuid,vbios_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None, None
    fields = [f.strip() for f in output.splitlines()[0].split(",", maxsplit=1)]
    gpu_uuid = fields[0] if len(fields) >= 1 else None
    gpu_vbios = fields[1] if len(fields) >= 2 else None
    return gpu_uuid, gpu_vbios


def _egl_info() -> dict[str, str | None]:
    info: dict[str, str | None] = {
        "egl_vendor": None,
        "egl_version": None,
        "egl_client_apis": None,
        "error": None,
    }
    try:
        import ctypes
        import ctypes.util as ctypes_util  # noqa: F811

        libegl_path = ctypes_util.find_library("EGL")
        if libegl_path is None:
            info["error"] = "libEGL not found via ctypes.util.find_library"
            return info
        libegl = ctypes.CDLL(libegl_path)

        EGL_NO_DISPLAY = ctypes.c_void_p(0)
        EGL_DEFAULT_DISPLAY = ctypes.c_void_p(1)

        eglGetDisplay = libegl.eglGetDisplay
        eglGetDisplay.restype = ctypes.c_void_p
        eglGetDisplay.argtypes = [ctypes.c_void_p]

        display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
        if not display or display == EGL_NO_DISPLAY:
            info["error"] = "eglGetDisplay returned null"
            return info

        eglInitialize = libegl.eglInitialize
        eglInitialize.restype = ctypes.c_int
        eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        major = ctypes.c_int()
        minor = ctypes.c_int()
        if not eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
            info["error"] = "eglInitialize failed"
            return info

        eglQueryString = libegl.eglQueryString
        eglQueryString.restype = ctypes.c_char_p
        eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]

        EGL_VENDOR = 0x3053
        EGL_VERSION = 0x3054
        EGL_CLIENT_APIS = 0x308D

        vendor = eglQueryString(display, EGL_VENDOR)
        version = eglQueryString(display, EGL_VERSION)
        client_apis = eglQueryString(display, EGL_CLIENT_APIS)
        info["egl_vendor"] = vendor.decode("utf-8") if vendor else None
        info["egl_version"] = version.decode("utf-8") if version else None
        info["egl_client_apis"] = client_apis.decode("utf-8") if client_apis else None
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _glx_info() -> dict[str, str | None]:
    info: dict[str, str | None] = {"found": None, "error": None}
    try:
        import ctypes
        import ctypes.util as ctypes_util  # noqa: F811

        libglx_path = ctypes_util.find_library("GLX")
        info["found"] = libglx_path
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _glvnd_info() -> dict[str, str | None]:
    info: dict[str, str | None] = {"found": None, "error": None}
    try:
        import ctypes
        import ctypes.util as ctypes_util  # noqa: F811

        lib_path = ctypes_util.find_library("GLdispatch")
        if lib_path is None:
            lib_path = ctypes_util.find_library("GL")
        info["found"] = lib_path
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _nvidia_egl_info() -> dict[str, str | None]:
    info: dict[str, str | None] = {"path": None, "error": None}
    try:
        for line in subprocess.check_output(
            ["ldconfig", "-p"], text=True, timeout=10
        ).splitlines():
            if "libEGL_nvidia" in line:
                info["path"] = line.strip()
                break
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _dpkg_info(*, package: str) -> str | None:
    output = _command_output(
        command=["dpkg", "-s", package],
    )
    if output is None:
        return None
    for line in output.splitlines():
        if line.startswith("Version:"):
            return line.partition(":")[2].strip()
    return None


def build_host_info(*, project_root: Path) -> dict[str, object]:
    git_commit = _command_output(
        command=["git", "rev-parse", "HEAD"],
        cwd=project_root,
    )
    git_status = _command_output(
        command=["git", "status", "--porcelain"],
        cwd=project_root,
    )

    gpu_uuid, gpu_vbios = _gpu_uuid_vbios()

    return {
        "hostname": os.uname().nodename,
        "cpu_model": _cpu_model(),
        "platform": os.uname().sysname,
        "kernel": os.uname().release,
        "gpu_uuid": gpu_uuid,
        "gpu_vbios": gpu_vbios,
        "opengl": _opengl_info(),
        "egl": _egl_info(),
        "glx": _glx_info(),
        "glvnd": _glvnd_info(),
        "nvidia_egl": _nvidia_egl_info(),
        "nvidia_libegl_version": _dpkg_info(package="libegl-nvidia0")
        or _dpkg_info(package="libnvidia-egl-wayland1")
        or _dpkg_info(package="libnvidia-eglcore"),
        "packages": _package_versions(),
        "git_commit": git_commit,
        "git_dirty": bool(git_status) if git_status is not None else None,
        "env_MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "env_CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def save_episode_diagnostic(
    *,
    out_dir: str | Path,
    episode_idx: int,
    obs: list[np.ndarray],
    actions: list[np.ndarray],
    img_obs: list[np.ndarray],
    qpos: list[np.ndarray],
    qvel: list[np.ndarray],
    ctrl: list[np.ndarray],
    body_xpos: list[np.ndarray],
    body_xmat: list[np.ndarray],
    sim_time: list[float],
    expert_phase: list[int],
    cube_init_pos: np.ndarray,
    tray_init_pos: np.ndarray,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_path = out_dir / f"pick_place_{episode_idx:06d}.npz"

    np.savez_compressed(
        episode_path,
        obs=np.asarray(obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        img_obs=np.asarray(img_obs, dtype=np.uint8),
        qpos=np.asarray(qpos, dtype=np.float64),
        qvel=np.asarray(qvel, dtype=np.float64),
        ctrl=np.asarray(ctrl, dtype=np.float64),
        body_xpos=np.asarray(body_xpos, dtype=np.float64),
        body_xmat=np.asarray(body_xmat, dtype=np.float64),
        sim_time=np.asarray(sim_time, dtype=np.float64),
        expert_phase=np.asarray(expert_phase, dtype=np.int8),
        cube_init_pos=np.asarray(cube_init_pos, dtype=np.float64),
        tray_init_pos=np.asarray(tray_init_pos, dtype=np.float64),
    )


def collect_diagnostic_episodes(
    *,
    episodes: int,
    out_dir: Path,
    seed: int,
    max_steps: int,
    capture_hz: float = DEFAULT_CAPTURE_HZ,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.npz")))
    if existing >= episodes:
        print(f"{existing} episodes already exist in {out_dir}, skipping collection.")
        return out_dir

    sim = SimEnv(randomize_scene=True, seed=seed)
    try:
        model = sim.model
        data = sim.data
        nbody = model.nbody

        progress = tqdm(range(episodes), desc="episodes", unit="episode")
        for episode_idx in progress:
            sim.reset_episode()
            cube_init_pos = data.body("cube").xpos.copy()
            tray_init_pos = data.site("tray_center").xpos.copy()

            controller = PickPlaceController(model, data)

            obs_list: list[np.ndarray] = []
            actions_list: list[np.ndarray] = []
            img_obs_list: list[np.ndarray] = []
            qpos_list: list[np.ndarray] = []
            qvel_list: list[np.ndarray] = []
            ctrl_list: list[np.ndarray] = []
            body_xpos_list: list[np.ndarray] = []
            body_xmat_list: list[np.ndarray] = []
            sim_time_list: list[float] = []
            phase_list: list[int] = []

            capture_period = 1.0 / capture_hz
            next_capture_time = float(data.time)

            step_count = 0
            step_progress = tqdm(
                range(max_steps),
                desc=f"ep {episode_idx:06d}",
                leave=False,
                unit="step",
            )
            for step_count, _ in enumerate(step_progress, start=1):
                capture_due = data.time + 1e-9 >= next_capture_time
                if capture_due:
                    obs_list.append(sim.build_observation())
                    img_obs_list.append(sim.build_image())
                    qpos_list.append(data.qpos.copy())
                    qvel_list.append(data.qvel.copy())
                    ctrl_list.append(data.ctrl.copy())
                    body_xpos_list.append(data.xpos[:nbody].copy())
                    body_xmat_list.append(data.xmat[:nbody].copy())
                    sim_time_list.append(float(data.time))
                    phase_list.append(controller.phase.value)

                controller.control()
                if capture_due:
                    actions_list.append(sim.build_action())
                    next_capture_time += capture_period

                mujoco.mj_step(model, data)

                controller.max_cube_z = max(
                    controller.max_cube_z,
                    float(data.body("cube").xpos[2]),
                )
                controller.update_phase()
                step_progress.set_postfix(phase=controller.phase.name)

                if controller.phase == Phase.DONE:
                    break

            if not obs_list:
                print(f"WARNING: episode {episode_idx} has 0 captures, skipping save")
                continue

            save_episode_diagnostic(
                out_dir=out_dir,
                episode_idx=episode_idx,
                obs=obs_list,
                actions=actions_list,
                img_obs=img_obs_list,
                qpos=qpos_list,
                qvel=qvel_list,
                ctrl=ctrl_list,
                body_xpos=body_xpos_list,
                body_xmat=body_xmat_list,
                sim_time=sim_time_list,
                expert_phase=phase_list,
                cube_init_pos=cube_init_pos,
                tray_init_pos=tray_init_pos,
            )
            progress.set_postfix(phase=controller.phase.name, steps=step_count)
    finally:
        sim.close()

    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-host reproducibility diagnostic: expert collection with float64 state."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--capture-hz", type=float, default=DEFAULT_CAPTURE_HZ)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=f"data/diagnostic/{timestamp}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_reproducibility(seed=args.seed)

    project_root = Path(__file__).resolve().parents[1]

    host_info = build_host_info(project_root=project_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    host_info_path = out_dir / "host_info.json"
    host_info_path.write_text(json.dumps(host_info, indent=2, sort_keys=True) + "\n")
    print(f"Host info written to {host_info_path}")

    print(f"\nCollecting {args.episodes} diagnostic episodes (seed={args.seed})...")
    collect_diagnostic_episodes(
        episodes=args.episodes,
        out_dir=out_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        capture_hz=args.capture_hz,
    )

    print(f"\nDiagnostic collection complete. Data in {out_dir}")


if __name__ == "__main__":
    main()
