from pathlib import Path
import sys
import tempfile

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rollout import save_dagger_episode, select_dagger_control  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    remaining = 0
    execute_expert = []
    gripper_disagreements = (False, False, False, False, False, True, False)
    for disagreement, gripper_disagreement in zip(
        (0.5, 1.1, 2.0, 2.0, 2.0, 0.5, 0.5),
        gripper_disagreements,
    ):
        execute, remaining = select_dagger_control(
            mode="threshold",
            beta=0.0,
            arm_disagreement=disagreement,
            gripper_disagreement=gripper_disagreement,
            intervention_threshold=1.0,
            intervention_steps=3,
            intervention_steps_remaining=remaining,
            rng=rng,
        )
        execute_expert.append(execute)
    assert execute_expert == [False, True, True, True, True, True, True]

    execute, remaining = select_dagger_control(
        mode="threshold",
        beta=0.0,
        arm_disagreement=0.5,
        gripper_disagreement=True,
        intervention_threshold=1.0,
        intervention_steps=3,
        intervention_steps_remaining=0,
        rng=rng,
    )
    assert execute
    assert remaining == 2

    with tempfile.TemporaryDirectory() as temp_dir:
        save_dagger_episode(
            dagger_dir=Path(temp_dir),
            episode_idx=0,
            beta=0.0,
            intervention_mode="threshold",
            intervention_threshold=1.0,
            intervention_steps=3,
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
            assert data["intervention_mode"].item() == "threshold"
            assert data["intervention_threshold"].item() == 1.0
            assert data["intervention_steps"].item() == 3

    print("DAgger metadata smoke test passed.")


if __name__ == "__main__":
    main()
