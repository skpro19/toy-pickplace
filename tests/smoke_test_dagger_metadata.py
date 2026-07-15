from pathlib import Path
import sys
import tempfile

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rollout import save_dagger_episode  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        save_dagger_episode(
            dagger_dir=Path(temp_dir),
            episode_idx=0,
            beta=0.5,
            result={
                "steps": 2,
                "seed": 123,
                "phases": [1, 2],
                "arm_disagreement": [0.1, 0.2],
                "gripper_disagreement": [False, True],
                "terminal_reason": "max_steps",
                "final_phase": 2,
                "cube_init_pos": np.zeros(3, dtype=np.float32),
                "tray_init_pos": np.zeros(3, dtype=np.float32),
                "log_buffers": {},
                "observations": [np.zeros(45, dtype=np.float32)] * 2,
                "expert_actions": [np.zeros(8, dtype=np.float32)] * 2,
                "policy_actions": [np.zeros(8, dtype=np.float32)] * 2,
                "executed_actions": [np.zeros(8, dtype=np.float32)] * 2,
                "expert_action_mask": [False, True],
                "task_metrics": {
                    "grasped": True,
                    "lifted": True,
                    "tray_reached": True,
                    "lowered_to_tray": False,
                    "released_over_tray": False,
                    "placement_success": False,
                },
            },
        )
        with np.load(Path(temp_dir) / "pick_place_000000.npz") as data:
            assert data["phases"].tolist() == [1, 2]
            assert np.allclose(data["arm_disagreement"], [0.1, 0.2])
            assert data["gripper_disagreement"].tolist() == [False, True]
            assert data["terminal_reason"].item() == "max_steps"
            assert data["final_phase"].item() == 2
            assert data["seed"].item() == 123
            assert data["grasped"].item()

    print("DAgger metadata smoke test passed.")


if __name__ == "__main__":
    main()
