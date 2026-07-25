from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from constant import ACTION_DIMS, OBS_DIMS  # noqa: E402
from data import DataCollector  # noqa: E402
from models.mlp import MLP  # noqa: E402
from policy_runtimes.registry import load_runtime  # noqa: E402
from rollout_core.episode import run_policy_episode  # noqa: E402
from sim import SimEnv  # noqa: E402


def save_test_checkpoint(*, path: Path) -> None:
    torch.manual_seed(0)
    model = MLP(obs_dim=OBS_DIMS, action_dim=ACTION_DIMS)
    torch.save(
        {
            "arch": "mlp",
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
    sim = SimEnv(randomize_scene=False, seed=0)
    try:
        sim.reset_episode()
        episode = DataCollector(sim=sim).collect_episode(
            max_steps=500,
            episode_idx=0,
            capture_hz=60.0,
        )
    finally:
        sim.close()

    assert episode["steps"] == 500
    assert len(episode["observations"]) == 60
    assert len(episode["actions"]) == 60
    print("60 Hz expert capture smoke test passed.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "best.pt"
        save_test_checkpoint(path=checkpoint_path)
        runtime = load_runtime(
            model_path=checkpoint_path,
            device=torch.device("cpu"),
        )

        rollout_sim = SimEnv(randomize_scene=False, seed=0)
        try:
            result = run_policy_episode(
                sim=rollout_sim,
                runtime=runtime,
                max_steps=500,
                capture_hz=60.0,
                log_rollout=True,
                rng=np.random.default_rng(0),
            )
        finally:
            rollout_sim.close()

    assert result["steps"] == 500
    assert len(result["log_buffers"]["rollout_obs"]) == 60
    print("60 Hz rollout capture smoke test passed.")


if __name__ == "__main__":
    main()
