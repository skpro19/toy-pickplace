"""Back up and restore flywheel training runs in an S3 bucket.

The tool mirrors checkpoints, TensorBoard runs, DAgger datasets, and results
under a session prefix while preserving their local project layout.

Examples:
    uv run python scripts/s3_backup.py upload --prefix 20260717-153000
    uv run python scripts/s3_backup.py list
    uv run python scripts/s3_backup.py download 20260717-153000/run-003
    uv run python scripts/s3_backup.py rm 20260717-153000

Configuration:
    S3_BUCKET          Required destination bucket unless --bucket is passed.
    S3_PREFIX          Optional key prefix within the bucket.
    S3_ENDPOINT_URL    Optional endpoint for an S3-compatible service.
    AWS_REGION         Optional AWS region.

Boto3's standard credential chain supplies credentials. This includes IAM
roles, AWS profiles, and the AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY variables.
"""

import argparse
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError


COMPONENT_MAP: dict[str, tuple[Path, str]] = {
    "checkpoints": (Path("checkpoints/flywheel"), "checkpoints"),
    "runs": (Path("runs/flywheel"), "runs"),
    "dagger": (Path("data/flywheel"), "data/flywheel"),
    "results": (Path("results/flywheel"), "results"),
}
COMPONENT_DEFAULT = list(COMPONENT_MAP)


def _s3_client(*, region: str | None, endpoint_url: str | None) -> Any:
    return boto3.client("s3", region_name=region, endpoint_url=endpoint_url)


def _selected_components(value: str) -> list[str]:
    selected = COMPONENT_DEFAULT if value == "all" else value.split(",")
    unknown = [name for name in selected if name not in COMPONENT_MAP]
    if unknown:
        raise ValueError(f"Unknown component(s): {', '.join(unknown)}")
    return selected


def _key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _validate_remote_path(path: str) -> str:
    normalized = path.strip("/")
    parts = PurePosixPath(normalized).parts
    if not normalized or any(part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid backup path: {path}")
    return normalized


def _list_objects(*, client: Any, bucket: str, prefix: str) -> dict[str, dict[str, Any]]:
    paginator = client.get_paginator("list_objects_v2")
    objects: dict[str, dict[str, Any]] = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.update({item["Key"]: item for item in page.get("Contents", [])})
    return objects


def _list_keys(*, client: Any, bucket: str, prefix: str) -> list[str]:
    return list(_list_objects(client=client, bucket=bucket, prefix=prefix))


def _delete_keys(*, client: Any, bucket: str, keys: list[str]) -> None:
    for start in range(0, len(keys), 1000):
        batch = keys[start : start + 1000]
        response = client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
        )
        errors = response.get("Errors", [])
        if errors:
            details = ", ".join(error.get("Key", "unknown") for error in errors)
            raise RuntimeError(f"Failed to delete S3 objects: {details}")


def _sync_directory(
    *,
    client: Any,
    bucket: str,
    source: Path,
    destination: str,
) -> tuple[int, int]:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    uploads = [
        (path, _key(destination, path.relative_to(source).as_posix()))
        for path in files
    ]
    expected_keys = {object_key for _, object_key in uploads}
    existing_objects = _list_objects(
        client=client,
        bucket=bucket,
        prefix=f"{destination}/",
    )
    uploaded = 0

    for path, object_key in uploads:
        remote = existing_objects.get(object_key)
        modified = remote.get("LastModified") if remote else None
        local = path.stat()
        if (
            remote
            and remote.get("Size") == local.st_size
            and modified is not None
            and modified.timestamp() >= local.st_mtime
        ):
            continue
        print(f"Uploading {path} -> s3://{bucket}/{object_key}")
        client.upload_file(str(path), bucket, object_key)
        uploaded += 1

    stale_keys = sorted(set(existing_objects) - expected_keys)
    _delete_keys(client=client, bucket=bucket, keys=stale_keys)
    return len(files), uploaded


def cmd_upload(args: argparse.Namespace) -> None:
    client = _s3_client(region=args.region, endpoint_url=args.endpoint_url)
    selected = _selected_components(args.components)
    discovered = 0
    uploaded = 0

    for name in selected:
        local_root, remote_dir = COMPONENT_MAP[name]
        source = local_root / args.run if args.run else local_root
        if not source.is_dir() or not any(
            path.is_file() for path in source.rglob("*")
        ):
            continue
        destination = _key(args.key_prefix, args.prefix, remote_dir, args.run or "")
        component_files, component_uploads = _sync_directory(
            client=client,
            bucket=args.bucket,
            source=source,
            destination=destination,
        )
        discovered += component_files
        uploaded += component_uploads

    if not discovered:
        raise RuntimeError("No data found for the selected components")
    print(f"Sync complete: {discovered} file(s), {uploaded} uploaded.")


def cmd_download(args: argparse.Namespace) -> None:
    client = _s3_client(region=args.region, endpoint_url=args.endpoint_url)
    backup_path = _validate_remote_path(args.path)
    parts = backup_path.split("/", 1)
    session = parts[0]
    run_name = parts[1] if len(parts) == 2 else None
    selected = _selected_components(args.components)
    output_root = Path(args.output)
    downloaded = 0

    for name in selected:
        local_root, remote_dir = COMPONENT_MAP[name]
        remote_base = _key(args.key_prefix, session, remote_dir)
        search_prefix = _key(remote_base, run_name or "") + "/"
        object_keys = _list_keys(
            client=client,
            bucket=args.bucket,
            prefix=search_prefix,
        )
        for object_key in object_keys:
            relative_key = object_key.removeprefix(f"{remote_base}/")
            relative_path = PurePosixPath(relative_key)
            if not relative_key or ".." in relative_path.parts:
                continue
            destination = (
                output_root / local_root / session / Path(*relative_path.parts)
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading s3://{args.bucket}/{object_key} -> {destination}")
            client.download_file(args.bucket, object_key, str(destination))
            downloaded += 1

    if not downloaded:
        raise RuntimeError(f"No files found for backup path {backup_path}")
    print(f"Downloaded {downloaded} file(s) from {backup_path}.")


def cmd_list(args: argparse.Namespace) -> None:
    client = _s3_client(region=args.region, endpoint_url=args.endpoint_url)
    root = _key(args.key_prefix)
    object_prefix = f"{root}/" if root else ""
    keys = _list_keys(client=client, bucket=args.bucket, prefix=object_prefix)
    component_pattern = "|".join(
        re.escape(remote) for _, remote in COMPONENT_MAP.values()
    )
    run_pattern = re.compile(
        rf"^(?P<session>[^/]+)/(?:{component_pattern})/(?P<run>run-\d+)/"
    )
    sessions: dict[str, set[str]] = {}

    for object_key in keys:
        relative_key = object_key.removeprefix(object_prefix)
        match = run_pattern.match(relative_key)
        if match:
            sessions.setdefault(match.group("session"), set()).add(
                match.group("run")
            )

    if not sessions:
        print("No backups found.")
        return
    for session in sorted(sessions, reverse=True):
        print(f"{session}/  [{', '.join(sorted(sessions[session]))}]")


def cmd_rm(args: argparse.Namespace) -> None:
    client = _s3_client(region=args.region, endpoint_url=args.endpoint_url)
    backup_path = _validate_remote_path(args.path)
    parts = backup_path.split("/", 1)
    session = parts[0]

    if len(parts) == 1:
        prefixes = [_key(args.key_prefix, session) + "/"]
    else:
        run_name = parts[1]
        prefixes = [
            _key(args.key_prefix, session, remote_dir, run_name) + "/"
            for _, remote_dir in COMPONENT_MAP.values()
        ]

    keys = sorted(
        {
            object_key
            for prefix in prefixes
            for object_key in _list_keys(
                client=client,
                bucket=args.bucket,
                prefix=prefix,
            )
        }
    )
    if not keys:
        raise RuntimeError(f"No files found for backup path {backup_path}")
    _delete_keys(client=client, bucket=args.bucket, keys=keys)
    print(f"Removed {backup_path} ({len(keys)} file(s)).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Back up and restore flywheel runs in S3"
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET"),
        help="S3 bucket (default: $S3_BUCKET)",
    )
    parser.add_argument(
        "--key-prefix",
        default=os.environ.get("S3_PREFIX", ""),
        help="S3 key prefix (default: $S3_PREFIX)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION"),
        help="AWS region (default: $AWS_REGION)",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("S3_ENDPOINT_URL"),
        help="S3-compatible endpoint (default: $S3_ENDPOINT_URL)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload = subparsers.add_parser(
        "upload",
        help="Upload and synchronize backup components",
    )
    upload.add_argument("--prefix", required=True, help="Backup session prefix")
    upload.add_argument(
        "--components",
        default="all",
        help="Comma-separated checkpoints,runs,dagger,results",
    )
    upload.add_argument("run", nargs="?", help="Specific run, such as run-003")

    download = subparsers.add_parser("download", help="Download backup components")
    download.add_argument("path", help="Session or session/run path")
    download.add_argument(
        "--components",
        default="all",
        help="Comma-separated checkpoints,runs,dagger,results",
    )
    download.add_argument("--output", default=".", help="Output directory")

    subparsers.add_parser("list", help="List backup sessions and runs")
    remove = subparsers.add_parser("rm", help="Remove a session or run")
    remove.add_argument("path", help="Session or session/run path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.bucket:
        parser.error("--bucket or S3_BUCKET is required")
    try:
        if args.command == "upload":
            cmd_upload(args)
        elif args.command == "download":
            cmd_download(args)
        elif args.command == "list":
            cmd_list(args)
        elif args.command == "rm":
            cmd_rm(args)
    except (BotoCoreError, ClientError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
