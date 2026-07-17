"""Backup and restore flywheel runs via HuggingFace Hub.

Environment:
  HF_TOKEN       HuggingFace API token (required for upload, optional for public downloads)
  HF_REPO_ID     Target repo (overrides the --repo default)

Usage:
  # Upload all local runs under a session prefix
  uv run python scripts/hf_backup.py upload --prefix 20260717-153000

  # Upload a specific run
  uv run python scripts/hf_backup.py upload --prefix 20260717-153000 run-003

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

from huggingface_hub import HfApi, snapshot_download


REPO_DEFAULT = "skpro19/toy-pickplace-flywheel"
CKPT_ROOT = Path("checkpoints/flywheel")
RUNS_ROOT = Path("runs/flywheel")


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
    srcs: list[tuple[Path, str]] = []

    if args.run:
        ckpt_src = CKPT_ROOT / args.run
        runs_src = RUNS_ROOT / args.run
        if ckpt_src.is_dir():
            srcs.append((ckpt_src, f"{prefix}/checkpoints/{args.run}"))
        if runs_src.is_dir():
            srcs.append((runs_src, f"{prefix}/runs/{args.run}"))
        if not srcs:
            print(f"No data found for run {args.run}", file=sys.stderr)
            sys.exit(1)
    else:
        if CKPT_ROOT.is_dir():
            srcs.append((CKPT_ROOT, f"{prefix}/checkpoints"))
        if RUNS_ROOT.is_dir():
            srcs.append((RUNS_ROOT, f"{prefix}/runs"))
        if not srcs:
            print("No flywheel data found in checkpoints/flywheel/ or runs/flywheel/", file=sys.stderr)
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

    output_root = Path(args.output or ".")
    tmp = Path(tempfile.mkdtemp())

    allow_patterns = [f"{prefix}/**"]
    try:
        snapshot_download(
            repo_id=args.repo,
            local_dir=str(tmp / "snap"),
            allow_patterns=allow_patterns,
            repo_type="model",
            token=args.token,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        shutil.rmtree(tmp)
        sys.exit(1)

    snap = tmp / "snap"

    def _restore(*, src_base: Path, dst_base: Path) -> None:
        if not src_base.is_dir():
            return
        if run_name:
            subdir = src_base / run_name
            if subdir.is_dir():
                dst = dst_base / run_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(subdir, dst, dirs_exist_ok=True)
        else:
            dst_base.mkdir(parents=True, exist_ok=True)
            for item in src_base.iterdir():
                shutil.copytree(item, dst_base / item.name, dirs_exist_ok=True)

    _restore(
        src_base=snap / prefix / "checkpoints",
        dst_base=output_root / "checkpoints" / "flywheel",
    )
    _restore(
        src_base=snap / prefix / "runs",
        dst_base=output_root / "runs" / "flywheel",
    )

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

    run_pattern = re.compile(r"^(?P<prefix>\d{8}-\d{6})/(?:checkpoints|runs)/(?P<run>run-\d+)/")
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

    up = sub.add_parser("upload", help="Upload checkpoints/runs to HF Hub")
    up.add_argument(
        "--prefix", required=True,
        help="Backup session prefix, e.g. $(date +%%Y%%m%%d-%%H%%M%%S)",
    )
    up.add_argument("run", nargs="?", help="Specific run (e.g. run-003); omit to upload all")

    dl = sub.add_parser("download", help="Download checkpoints/runs from HF Hub")
    dl.add_argument("path", help="Path, e.g. 20260717-153000 or 20260717-153000/run-003")
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
