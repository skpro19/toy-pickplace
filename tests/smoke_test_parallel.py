from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from constant import ACTION_DIMS, OBS_DIMS  # noqa: E402
from eval import EVAL_METRIC_VERSION, score_ckpt, score_task_metrics  # noqa: E402
from mlp import MLP  # noqa: E402
from rollout import make_episode_seeds, rollout  # noqa: E402


def save_test_checkpoint(*, path: Path) -> None:
    torch.manual_seed(0)
    model = MLP(obs_dim=OBS_DIMS, action_dim=ACTION_DIMS)
    torch.save(
        {
            "model_dict": model.state_dict(),
            "normalize": True,
            "action_space": "joint_delta",
            "epoch": 1,
            "eval_score": 0.0,
            "arm_actions_mean": np.zeros((1, ACTION_DIMS - 1), dtype=np.float32),
            "arm_actions_std": np.ones((1, ACTION_DIMS - 1), dtype=np.float32),
            "arm_obs_mean": np.zeros((1, OBS_DIMS), dtype=np.float32),
            "arm_obs_std": np.ones((1, OBS_DIMS), dtype=np.float32),
        },
        path,
    )


def main() -> None:
    assert np.isclose(
        score_task_metrics(
            metrics={
                "grasped": True,
                "lifted": True,
                "tray_reached": True,
                "lowered_to_tray": True,
                "released_over_tray": False,
                "placement_success": False,
            }
        ),
        0.6,
    )
    assert np.isclose(
        score_task_metrics(
            metrics={
                "grasped": True,
                "lifted": True,
                "tray_reached": True,
                "lowered_to_tray": True,
                "released_over_tray": True,
                "placement_success": False,
            }
        ),
        0.7,
    )
    assert np.isclose(
        score_task_metrics(
            metrics={
                "grasped": True,
                "lifted": True,
                "tray_reached": True,
                "lowered_to_tray": True,
                "released_over_tray": True,
                "placement_success": True,
            }
        ),
        1.0,
    )
    assert make_episode_seeds(seed=42, episodes=3) == make_episode_seeds(
        seed=42,
        episodes=3,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        checkpoint_path = root / "policy.pt"
        dagger_dir = root / "dagger"
        save_test_checkpoint(path=checkpoint_path)

        scores = score_ckpt(
            ckpt_path=str(checkpoint_path),
            seed=42,
            max_steps=5,
            episodes=2,
            workers=2,
        )
        assert len(scores["scores"]) == 2
        assert scores["eval_metric_version"] == EVAL_METRIC_VERSION
        for key in (
            "mean_score",
            "grasp_rate",
            "lift_rate",
            "tray_reach_rate",
            "lowered_to_tray_rate",
            "released_over_tray_rate",
            "placement_success_rate",
        ):
            assert 0.0 <= scores[key] <= 1.0

        serial_scores = score_ckpt(
            ckpt_path=str(checkpoint_path),
            seed=42,
            max_steps=5,
            episodes=2,
            workers=1,
        )
        assert serial_scores == scores

        expert_scores = score_ckpt(
            ckpt_path=str(checkpoint_path),
            seed=42,
            max_steps=1400,
            episodes=1,
            expert_baseline=True,
            workers=2,
        )
        expert_serial_scores = score_ckpt(
            ckpt_path=str(checkpoint_path),
            seed=42,
            max_steps=1400,
            episodes=1,
            expert_baseline=True,
            workers=1,
        )
        assert expert_scores == expert_serial_scores
        assert expert_scores["lowered_to_tray_rate"] == 1.0
        assert expert_scores["released_over_tray_rate"] == 1.0
        assert expert_scores["placement_success_rate"] == 1.0

        rollout(
            model_path=str(checkpoint_path),
            randomize_scene=True,
            seed=42,
            episodes=2,
            max_steps=5,
            log_root=None,
            train_npz_dir=None,
            log_rollout=False,
            dagger=True,
            dagger_root=dagger_dir,
            beta=0.5,
            create_dagger_subdir=False,
            headless=True,
            workers=2,
        )
        output_names = sorted(path.name for path in dagger_dir.glob("*.npz"))
        assert output_names == ["pick_place_000000.npz", "pick_place_000001.npz"]

    print("Parallel evaluation and DAgger smoke test passed.")


if __name__ == "__main__":
    main()
