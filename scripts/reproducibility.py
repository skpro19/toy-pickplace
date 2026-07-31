import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
ENVIRONMENT_MANIFEST_SCHEMA_VERSION = 1


def configure_reproducibility(*, seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")

    configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_initialized() and configured_workspace != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be set before CUDA is initialized"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _command_output(*, command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None
    return result.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    package_names = ("glfw", "mink", "mujoco", "numpy", "PyOpenGL", "torch")
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _cpu_model() -> str | None:
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.is_file():
        for line in cpuinfo_path.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or None


def _nvidia_gpus() -> list[dict[str, str]]:
    output = _command_output(
        command=[
            "nvidia-smi",
            "--query-gpu=index,name,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []

    gpus = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) != 3:
            continue
        index, name, driver_version = fields
        gpus.append(
            {
                "index": index,
                "name": name,
                "driver_version": driver_version,
            }
        )
    return gpus


def _opengl_info() -> dict[str, str | None]:
    info: dict[str, str | None] = {
        "backend": os.environ.get("MUJOCO_GL"),
        "vendor": None,
        "renderer": None,
        "version": None,
        "error": None,
    }
    context = None
    try:
        import mujoco
        from OpenGL import GL

        context = mujoco.GLContext(1, 1)
        context.make_current()
        values = {
            "vendor": GL.glGetString(GL.GL_VENDOR),
            "renderer": GL.glGetString(GL.GL_RENDERER),
            "version": GL.glGetString(GL.GL_VERSION),
        }
        for key, value in values.items():
            info[key] = value.decode("utf-8") if value is not None else None
    except Exception as error:
        info["error"] = f"{type(error).__name__}: {error}"
    finally:
        if context is not None:
            try:
                context.free()
            except Exception as error:
                if info["error"] is None:
                    info["error"] = f"{type(error).__name__}: {error}"
    return info


def build_environment_manifest(
    *,
    run_name: str,
    arch: str,
    project_root: Path,
) -> dict[str, object]:
    lock_path = project_root / "uv.lock"
    git_commit = _command_output(
        command=["git", "rev-parse", "HEAD"],
        cwd=project_root,
    )
    git_status = _command_output(
        command=["git", "status", "--porcelain"],
        cwd=project_root,
    )

    return {
        "schema_version": ENVIRONMENT_MANIFEST_SCHEMA_VERSION,
        "run_name": run_name,
        "arch": arch,
        "git": {
            "commit": git_commit,
            "dirty": bool(git_status) if git_status is not None else None,
        },
        "system": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "python_version": platform.python_version(),
        },
        "software": {
            "packages": _package_versions(),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "uv": _command_output(command=["uv", "--version"]),
            "uv_lock_sha256": (
                hashlib.sha256(lock_path.read_bytes()).hexdigest()
                if lock_path.is_file()
                else None
            ),
        },
        "nvidia_gpus": _nvidia_gpus(),
        "opengl": _opengl_info(),
        "reproducibility": {
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def write_environment_manifest(
    *,
    path: Path,
    manifest: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)
