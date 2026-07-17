"""Backup and restore flywheel runs via HuggingFace Hub.

Environment:
  HF_TOKEN       HuggingFace API token (required for upload, optional for public downloads)
  HF_REPO_ID     Target repo (overrides the --repo default)

Components (selectable via --components):
  checkpoints    Model weights (.pt files) from checkpoints/flywheel/
  runs           TensorBoard event logs from runs/flywheel/
  dagger         DAgger datasets from data/flywheel/
  results        Evaluation scores from results/flywheel/

Usage:
  # Upload all components for all local runs
  uv run python scripts/hf_backup.py upload --prefix 20260717-153000

  # Upload only checkpoints and dagger for a specific run
  uv run python scripts/hf_backup.py upload --prefix 20260717-153000 --components checkpoints,dagger run-003

  # List available sessions and runs in the repo
  uv run python scripts/hf_backup.py list

  # Download everything from a session
  uv run python scripts/hf_backup.py download 20260717-153000

  # Download a specific run from a session
  uv run python scripts/hf_backup.py download 20260717-153000/run-003

  # Remove a session or specific run from the repo
  uv run python scripts/hf_backup.py rm 20260717-153000
  uv run python scripts/hf_backup.py rm 20260717-153000/run-003
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


REPO_DEFAULT = "skpro19/toy-pickplace-flywheel"
CKPT_ROOT = Path("checkpoints/flywheel")
RUNS_ROOT = Path("runs/flywheel")
DAGGER_ROOT = Path("data/flywheel")
RESULTS_ROOT = Path("results/flywheel")

COMPONENT_MAP: dict[str, tuple[Path, str]] = {
    "checkpoints": (CKPT_ROOT, "checkpoints"),
    "runs": (RUNS_ROOT, "runs"),
    "dagger": (DAGGER_ROOT, "data/flywheel"),
    "results": (RESULTS_ROOT, "results"),
}
COMPONENT_DEFAULT = list(COMPONENT_MAP)

# Remote subdirectory under the session prefix for each component.
# Used by both upload remote_dir and download restore paths.
COMPONENT_REMOTE: dict[str, str] = {n: p[1] for n, p in COMPONENT_MAP.items()}


def _api(*, token: Optional[str] = None) -> HfApi:
    return HfApi(token=token)


def _ensure_repo(*, api: HfApi, repo_id: str) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception:
        print(f"Creating repo {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)


def cmd_upload(args: argparse.Namespace) -> None:
    api = _api(token=args.token)
    _ensure_repo(api=api, repo_id=args.repo)

    prefix = args.prefix
    selected = COMPONENT_DEFAULT if args.components == "all" else args.components.split(",")
    srcs: list[tuple[Path, str]] = []

    for name in selected:
        if name not in COMPONENT_MAP:
            print(f"Unknown component: {name}", file=sys.stderr)
            sys.exit(1)
        local_root, remote_dir = COMPONENT_MAP[name]
        if args.run:
            src = local_root / args.run
            if src.is_dir():
                srcs.append((src, f"{prefix}/{remote_dir}/{args.run}"))
        else:
            if local_root.is_dir():
                srcs.append((local_root, f"{prefix}/{remote_dir}"))

    if not srcs:
        print("No data found for the selected components", file=sys.stderr)
        sys.exit(1)

    for src, dst in srcs:
        print(f"Uploading {src}/ -> {args.repo}/{dst}/")
        api.upload_folder(
            repo_id=args.repo,
            folder_path=str(src),
            path_in_repo=dst,
            repo_type="model",
            delete_patterns=[f"{dst}/*"],
        )
    print("Upload complete.")


def cmd_download(args: argparse.Namespace) -> None:
    api = _api(token=args.token)

    parts = args.path.split("/", 1)
    prefix = parts[0]
    run_name = parts[1] if len(parts) > 1 else None

    selected = COMPONENT_DEFAULT if args.components == "all" else args.components.split(",")
    selected_remotes = [COMPONENT_REMOTE[n] for n in selected if n in COMPONENT_REMOTE]

    if not selected_remotes:
        print("No valid components selected", file=sys.stderr)
        sys.exit(1)

    output_root = Path(args.output or ".")
    tmp = Path(tempfile.mkdtemp())

    allow_patterns = [f"{prefix}/{r}/**" for r in selected_remotes]
    success = False
    try:
        snapshot_download(
            repo_id=args.repo,
            local_dir=str(tmp / "snap"),
            allow_patterns=allow_patterns,
            repo_type="model",
            token=args.token,
            local_dir_use_symlinks=False,
        )
        success = True
    except Exception as e:
        print(f"snapshot_download failed ({e}), falling back to individual downloads...", file=sys.stderr)

    if not success:
        snap = tmp / "snap"
        try:
            files = api.list_repo_files(repo_id=args.repo, repo_type="model")
        except Exception as e2:
            print(f"Failed to list repo: {e2}", file=sys.stderr)
            shutil.rmtree(tmp)
            sys.exit(1)

        matched = [f for f in files if any(f.startswith(f"{prefix}/{r}/") for r in selected_remotes)]
        if not matched:
            print(f"No files found for prefix {prefix}", file=sys.stderr)
            shutil.rmtree(tmp)
            sys.exit(1)

        for f in matched:
            dest = snap / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(
                    repo_id=args.repo,
                    filename=f,
                    local_dir=str(snap),
                    repo_type="model",
                    token=args.token,
                    local_dir_use_symlinks=False,
                )
            except Exception as e3:
                print(f"  WARNING: could not download {f}: {e3}", file=sys.stderr)

    snap = tmp / "snap"

    def _restore(*, src_base: Path, dst_base: Path) -> None:
        if not src_base.is_dir():
            return
        if run_name:
            subdir = src_base / run_name
            if subdir.is_dir():
                dst = dst_base / prefix / run_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(subdir, dst, dirs_exist_ok=True)
        else:
            dst_base.mkdir(parents=True, exist_ok=True)
            for item in src_base.iterdir():
                shutil.copytree(item, dst_base / prefix / item.name, dirs_exist_ok=True)

    # Restore only the selected components
    _restore(
        src_base=snap / prefix / "checkpoints",
        dst_base=output_root / "checkpoints" / "flywheel",
    ) if "checkpoints" in selected else None
    _restore(
        src_base=snap / prefix / "runs",
        dst_base=output_root / "runs" / "flywheel",
    ) if "runs" in selected else None
    _restore(
        src_base=snap / prefix / "data" / "flywheel",
        dst_base=output_root / "data" / "flywheel",
    ) if "dagger" in selected else None
    _restore(
        src_base=snap / prefix / "results",
        dst_base=output_root / "results" / "flywheel",
    ) if "results" in selected else None

    shutil.rmtree(tmp)
    print(f"Downloaded {args.path}")


def cmd_list(args: argparse.Namespace) -> None:
    api = _api(token=args.token)
    try:
        files = api.list_repo_files(repo_id=args.repo, repo_type="model")
    except Exception as e:
        print(f"Failed to list repo: {e}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("No backups found.")
        return

    run_pattern = re.compile(r"^(?P<prefix>\d{8}-\d{6})/(?:checkpoints|runs|data/flywheel|results)/(?P<run>run-\d+)/")
    sessions: dict[str, set[str]] = {}
    for f in files:
        m = run_pattern.match(f)
        if m:
            sessions.setdefault(m.group("prefix"), set()).add(m.group("run"))

    if not sessions:
        print("No backups found.")
        return

    for prefix in sorted(sessions, reverse=True):
        runs = sorted(sessions[prefix])
        runs_str = ", ".join(runs) if runs else "(empty)"
        print(f"{prefix}/  [{runs_str}]")


def cmd_rm(args: argparse.Namespace) -> None:
    api = _api(token=args.token)
    path = args.path.rstrip("/")
    try:
        api.delete_folder(
            repo_id=args.repo,
            path_in_repo=path,
            repo_type="model",
        )
        print(f"Removed {path}")
    except Exception as e:
        print(f"Failed to remove {path}: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backup and restore flywheel runs via HuggingFace Hub"
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("HF_REPO_ID", REPO_DEFAULT),
        help=f"HF repo ID (default: {REPO_DEFAULT}, or $HF_REPO_ID)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF API token (default: $HF_TOKEN)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="Upload checkpoints/runs/dagger/results to HF Hub")
    up.add_argument(
        "--prefix", required=True,
        help="Backup session prefix, e.g. $(date +%%Y%%m%%d-%%H%%M%%S)",
    )
    up.add_argument(
        "--components", default="all",
        help="Comma-separated components to upload: checkpoints,runs,dagger,results (default: all)",
    )
    up.add_argument("run", nargs="?", help="Specific run (e.g. run-003); omit to upload all")

    dl = sub.add_parser("download", help="Download checkpoints/runs/dagger/results from HF Hub")
    dl.add_argument("path", help="Path, e.g. 20260717-153000 or 20260717-153000/run-003")
    dl.add_argument(
        "--components", default="all",
        help="Comma-separated components to download: checkpoints,runs,dagger,results (default: all)",
    )
    dl.add_argument("--output", default=".", help="Output directory (default: current dir)")

    _ = sub.add_parser("list", help="List available backups in the repo")

    rm = sub.add_parser("rm", help="Remove a backup from the repo")
    rm.add_argument("path", help="Path to remove, e.g. 20260717-153000/run-003")

    args = parser.parse_args()

    if args.command == "upload":
        cmd_upload(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "rm":
        cmd_rm(args)


if __name__ == "__main__":
    main()
