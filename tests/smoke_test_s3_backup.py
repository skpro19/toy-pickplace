import argparse
import contextlib
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import s3_backup


class FakePaginator:
    def __init__(self, *, client: "FakeS3Client") -> None:
        self.client = client

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
        del Bucket
        contents = [
            {
                "Key": key,
                "Size": len(value),
                "LastModified": self.client.modified[key],
            }
            for key, value in sorted(self.client.objects.items())
            if key.startswith(Prefix)
        ]
        return [{"Contents": contents}] if contents else [{}]


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.modified: dict[str, datetime] = {}
        self.upload_count = 0

    def get_paginator(self, operation: str) -> FakePaginator:
        assert operation == "list_objects_v2"
        return FakePaginator(client=self)

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        del bucket
        self.objects[key] = Path(filename).read_bytes()
        self.modified[key] = datetime.now(timezone.utc)
        self.upload_count += 1

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        del bucket
        Path(filename).write_bytes(self.objects[key])

    def delete_objects(self, *, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:
        del Bucket
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)
            self.modified.pop(item["Key"], None)
        return {}


def args(**overrides: Any) -> argparse.Namespace:
    values = {
        "bucket": "test-bucket",
        "key_prefix": "toy-pickplace/flywheel",
        "region": None,
        "endpoint_url": None,
        "components": "all",
        "run": None,
        "prefix": "ablation-ratio-seed1-20260726-120000",
        "path": "ablation-ratio-seed1-20260726-120000",
        "output": ".",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def main() -> None:
    client = FakeS3Client()
    original_client_factory = s3_backup._s3_client
    s3_backup._s3_client = lambda **_: client

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_directory = Path.cwd()
            os.chdir(temp_dir)
            try:
                checkpoint = Path("checkpoints/flywheel/run-001/round-000/best.pt")
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint-v1")
                result = Path("results/flywheel/run-001/metrics.json")
                result.parent.mkdir(parents=True)
                result.write_text("{}")
                stale_key = (
                    "toy-pickplace/flywheel/ablation-ratio-seed1-20260726-120000/"
                    "checkpoints/stale.pt"
                )
                client.objects[stale_key] = b"stale"
                client.modified[stale_key] = datetime.now(timezone.utc)

                s3_backup.cmd_upload(args(components="checkpoints,results"))
                checkpoint_key = (
                    "toy-pickplace/flywheel/ablation-ratio-seed1-20260726-120000/"
                    "checkpoints/run-001/round-000/best.pt"
                )
                result_key = (
                    "toy-pickplace/flywheel/ablation-ratio-seed1-20260726-120000/"
                    "results/run-001/metrics.json"
                )
                assert client.objects[checkpoint_key] == b"checkpoint-v1"
                assert client.objects[result_key] == b"{}"
                assert stale_key not in client.objects
                assert client.upload_count == 2

                s3_backup.cmd_upload(args(components="checkpoints,results"))
                assert client.upload_count == 2

                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    s3_backup.cmd_list(args())
                assert "ablation-ratio-seed1-20260726-120000/  [run-001]" in output.getvalue()

                checkpoint.unlink()
                result.unlink()
                restore_root = Path("restore")
                s3_backup.cmd_download(
                    args(
                        path="ablation-ratio-seed1-20260726-120000/run-001",
                        components="checkpoints,results",
                        output=str(restore_root),
                    )
                )
                restored_checkpoint = (
                    restore_root
                    / "checkpoints/flywheel/ablation-ratio-seed1-20260726-120000"
                    / "run-001/round-000/best.pt"
                )
                assert restored_checkpoint.read_bytes() == b"checkpoint-v1"

                s3_backup.cmd_rm(
                    args(path="ablation-ratio-seed1-20260726-120000/run-001")
                )
                assert checkpoint_key not in client.objects
                assert result_key not in client.objects
            finally:
                os.chdir(original_directory)
    finally:
        s3_backup._s3_client = original_client_factory

    print("S3 backup smoke test passed")


if __name__ == "__main__":
    main()
