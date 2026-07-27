"""Back up and restore flywheel training runs in an S3 bucket.

The tool mirrors checkpoints, TensorBoard runs, DAgger datasets, and results
using the exact local project-relative paths as S3 object keys.

Examples:
    uv run python scripts/s3_backup.py upload run-name
    uv run python scripts/s3_backup.py list
    uv run python scripts/s3_backup.py download run-name
    uv run python scripts/s3_backup.py rm run-name

Configuration:
    S3_BUCKET          Required destination bucket unless --bucket is passed.
    S3_ENDPOINT_URL    Optional endpoint for an S3-compatible service.
    AWS_REGION         Optional AWS region.

Boto3's standard credential chain supplies credentials. This includes IAM
roles, AWS profiles, and the AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY variables.

Object keys use the exact local project-relative paths (e.g.,
``checkpoints/flywheel/<run-name>/round-000/best.pt``).
"""

import argparse
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)


COMPONENT_MAP: dict[str, tuple[Path, str]] = {
    "checkpoints": (Path("checkpoints/flywheel"), "checkpoints/flywheel"),
    "runs": (Path("runs/flywheel"), "runs/flywheel"),
    "dagger": (Path("data/flywheel"), "data/flywheel"),
    "results": (Path("results/flywheel"), "results/flywheel"),
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
    if not args.run:
        raise ValueError("A run name is required for upload")
    run_name = _validate_remote_path(args.run)
    client = _s3_client(region=args.region, endpoint_url=args.endpoint_url)
    selected = _selected_components(args.components)
    discovered = 0
    uploaded = 0

    for name in selected:
        local_root, remote_dir = COMPONENT_MAP[name]
        source = local_root / run_name
        if not source.is_dir() or not any(
            path.is_file() for path in source.rglob("*")
        ):
            continue
        destination = _key(remote_dir, run_name)
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
    run_name = _validate_remote_path(args.path)
    selected = _selected_components(args.components)
    output_root = Path(args.output)
    downloaded = 0

    for name in selected:
        local_root, remote_dir = COMPONENT_MAP[name]
        remote_base = _key(remote_dir)
        search_prefix = _key(remote_base, run_name) + "/"
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
                output_root / local_root / Path(*relative_path.parts)
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading s3://{args.bucket}/{object_key} -> {destination}")
            client.download_file(args.bucket, object_key, str(destination))
            downloaded += 1

    if not downloaded:
        raise RuntimeError(f"No files found for run {run_name}")
    print(f"Downloaded {downloaded} file(s) from {run_name}.")


def cmd_has_files(args: argparse.Namespace) -> None:
    client = _s3_client(region=args.region, endpoint_url=args.endpoint_url)
    run_name = _validate_remote_path(args.path)
    selected = _selected_components(args.components)

    for filename in args.files:
        relative_path = PurePosixPath(filename)
        if relative_path.is_absolute() or any(
            part in {".", ".."} for part in relative_path.parts
        ):
            raise ValueError(f"Invalid relative file path: {filename}")

        for name in selected:
            _, remote_dir = COMPONENT_MAP[name]
            object_key = _key(
                remote_dir,
                run_name,
                relative_path.as_posix(),
            )
            try:
                client.head_object(Bucket=args.bucket, Key=object_key)
            except ClientError as error:
                error_code = error.response.get("Error", {}).get("Code", "")
                if error_code in {"404", "NoSuchKey", "NotFound"}:
                    print(f"Missing: s3://{args.bucket}/{object_key}")
                    raise SystemExit(1) from error
                raise

    print(f"All requested files exist for {run_name}.")


def cmd_list(args: argparse.Namespace) -> None:
    client = _s3_client(region=args.region, endpoint_url=args.endpoint_url)
    component_pattern = "|".join(
        re.escape(remote) for _, remote in COMPONENT_MAP.values()
    )
    run_pattern = re.compile(
        rf"^(?:{component_pattern})/(?P<run>[^/]+)/"
    )
    runs: set[str] = set()

    for object_key in _list_keys(client=client, bucket=args.bucket, prefix=""):
        match = run_pattern.match(object_key)
        if match:
            runs.add(match.group("run"))

    if not runs:
        print("No backups found.")
        return
    for run in sorted(runs, reverse=True):
        print(run)


def cmd_rm(args: argparse.Namespace) -> None:
    client = _s3_client(region=args.region, endpoint_url=args.endpoint_url)
    run_name = _validate_remote_path(args.path)
    prefixes = [
        _key(remote_dir, run_name) + "/"
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
        raise RuntimeError(f"No files found for run {run_name}")
    _delete_keys(client=client, bucket=args.bucket, keys=keys)
    print(f"Removed {run_name} ({len(keys)} file(s)).")


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
        help=argparse.SUPPRESS,
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
    upload.add_argument("--prefix", default="", help="Backup session prefix (deprecated)")
    upload.add_argument(
        "--components",
        default="all",
        help="Comma-separated checkpoints,runs,dagger,results",
    )
    upload.add_argument("run", help="Specific run, such as run-003")

    download = subparsers.add_parser("download", help="Download backup components")
    download.add_argument("path", help="Run name to download")
    download.add_argument(
        "--components",
        default="all",
        help="Comma-separated checkpoints,runs,dagger,results",
    )
    download.add_argument("--output", default=".", help="Output directory")

    has_files = subparsers.add_parser(
        "has-files",
        help="Check whether files exist for a backed-up run",
    )
    has_files.add_argument("path", help="Run name to check")
    has_files.add_argument("files", nargs="+", help="Files relative to the run directory")
    has_files.add_argument(
        "--components",
        default="all",
        help="Comma-separated checkpoints,runs,dagger,results",
    )

    subparsers.add_parser("list", help="List backup sessions and runs")
    remove = subparsers.add_parser("rm", help="Remove a session or run")
    remove.add_argument("path", help="Run name to remove")
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
        elif args.command == "has-files":
            cmd_has_files(args)
        elif args.command == "list":
            cmd_list(args)
        elif args.command == "rm":
            cmd_rm(args)
    except ClientError as error:
        error_code = error.response.get("Error", {}).get("Code", "")
        auth_codes = {
            "AccessDenied",
            "ExpiredToken",
            "InvalidAccessKeyId",
            "InvalidClientTokenId",
            "SignatureDoesNotMatch",
            "UnrecognizedClientException",
        }
        exit_code = 2 if error_code in auth_codes else 1
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(exit_code) from error
    except (
        EndpointConnectionError,
        ConnectionClosedError,
        ConnectTimeoutError,
        ReadTimeoutError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(3) from error
    except (BotoCoreError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
