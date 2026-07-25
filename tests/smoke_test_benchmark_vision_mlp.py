import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = (
    Path(__file__).parents[1]
    / ".opencode/commands/scripts/benchmark-vision-mlp"
)
sys.path.insert(0, str(TOOLS_DIR))

import analyze
import capture_hardware
import validate_result
import verify_scripts


class BenchmarkVisionMlpTest(unittest.TestCase):
    def test_initial_plans_have_exact_matrix(self) -> None:
        training = analyze.expected_training_rows(batch_size=768)
        workers = analyze.expected_worker_rows()

        self.assertEqual(len(training), 21)
        self.assertEqual(len({row["trial_id"] for row in training}), 21)
        self.assertEqual(
            {row["trial_id"] for row in training if row["repetition"] == "3"},
            {
                "dw0-p0-r3",
                "dw2-p0-r3",
                "dw2-p1-r3",
                "dw4-p0-r3",
                "dw4-p1-r3",
                "dw8-p0-r3",
                "dw8-p1-r3",
            },
        )
        self.assertEqual(len(workers), 12)
        self.assertEqual(len({row["trial_id"] for row in workers}), 12)

    def test_stability_plans_add_two_repetitions(self) -> None:
        training = analyze.expected_stability_training(
            needed=["dw4-p0", "dw8-p1"], batch_size=768
        )
        workers = analyze.expected_stability_workers(needed=["6"])

        self.assertEqual(
            [row["trial_id"] for row in training],
            ["dw4-p0-r4", "dw4-p0-r5", "dw8-p1-r4", "dw8-p1-r5"],
        )
        self.assertEqual(
            [row["trial_id"] for row in workers],
            [
                "evaluation-w6-r4",
                "dagger-w6-r4",
                "evaluation-w6-r5",
                "dagger-w6-r5",
            ],
        )

    def test_downloaded_absolute_path_is_remapped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control_dir = Path(temporary)
            checkpoint = control_dir / "bootstrap/attempt-1/checkpoints/run/last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")

            resolved = validate_result.resolve_control_path(
                path_value=(
                    "/workspace/toy-pickplace/.benchmark/run/"
                    "bootstrap/attempt-1/checkpoints/run/last.pt"
                ),
                control_dir=control_dir,
                anchor="bootstrap",
            )

            self.assertEqual(resolved, checkpoint)

    def test_downloaded_path_wins_over_existing_remote_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control_dir = root / "download"
            local = control_dir / "bootstrap/run/last.pt"
            remote = root / "remote/bootstrap/run/last.pt"
            local.parent.mkdir(parents=True)
            remote.parent.mkdir(parents=True)
            local.write_bytes(b"downloaded")
            remote.write_bytes(b"unrelated")

            resolved = validate_result.resolve_control_path(
                path_value=str(remote),
                control_dir=control_dir,
                anchor="bootstrap",
            )

            self.assertEqual(resolved, local)

    def test_success_must_be_json_boolean(self) -> None:
        result = {
            "schema_version": 1,
            "success": 1,
            "attempt": 1,
        }

        with self.assertRaises(ValueError):
            validate_result.validate_common(
                result=result,
                metadata={"commit": "commit"},
                manifest={"aggregate_sha256": "digest"},
                kind="training",
                trial_id="trial",
                generation="generation",
                attempt=1,
            )

    def test_offer_contract_rejects_relaxed_hardware(self) -> None:
        offer = {
            "gpu_name": "RTX_4090",
            "gpu_frac": 1,
            "num_gpus": 1,
            "gpu_ram": 24564,
            "gpu_max_power": 400,
            "compute_cap": 890,
            "total_flops": 82,
            "cpu_cores_effective": 24,
            "cpu_ram": 65536,
            "disk_bw": 1000,
            "pci_gen": 4,
            "pcie_bw": 19,
            "inet_down": 500,
            "inet_up": 200,
            "reliability": 0.99,
            "rentable": True,
            "verification": "verified",
            "gpu_display_active": False,
        }

        failures = capture_hardware.require_offer_thresholds(offer=offer)

        self.assertIn("offer pcie_bw=19.0 is below 20", failures)

    def test_script_manifest_detects_changed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools_dir = root / "tools"
            tools_dir.mkdir()
            tool = tools_dir / "tool.py"
            tool.write_text("original\n")
            manifest = {"tool.py": "incorrect"}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaises(ValueError):
                verify_scripts.verify_scripts(
                    tools_dir=tools_dir, manifest=manifest_path
                )

            manifest["tool.py"] = hashlib.sha256(tool.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            (tools_dir / "extra.py").write_text("extra\n")
            with self.assertRaises(ValueError):
                verify_scripts.verify_scripts(
                    tools_dir=tools_dir, manifest=manifest_path
                )


if __name__ == "__main__":
    unittest.main()
