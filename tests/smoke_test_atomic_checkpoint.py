from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_core import checkpoint as checkpoint_module


def main() -> None:
    model = torch.nn.Linear(2, 1)
    norm_stats = {
        "arm_actions_mean": np.zeros(1),
        "arm_actions_std": np.ones(1),
        "arm_obs_mean": np.zeros(2),
        "arm_obs_std": np.ones(2),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "last.pt"
        model_path.write_bytes(b"previous-checkpoint")
        original_torch_save = checkpoint_module.torch.save

        def interrupted_save(value: object, path: Path) -> None:
            del value
            path.write_bytes(b"partial-checkpoint")
            raise RuntimeError("simulated interrupted save")

        checkpoint_module.torch.save = interrupted_save
        try:
            try:
                checkpoint_module.save_checkpoint(
                    arch="mlp",
                    model=model,
                    model_path=model_path,
                    epoch_number=1,
                    dropout=0.0,
                    normalize=True,
                    action_space="joint_delta",
                    norm_stats=norm_stats,
                )
            except RuntimeError as error:
                assert str(error) == "simulated interrupted save"
            else:
                raise AssertionError("Interrupted checkpoint save must fail")
        finally:
            checkpoint_module.torch.save = original_torch_save

        assert model_path.read_bytes() == b"previous-checkpoint"
        assert list(model_path.parent.glob("*.tmp")) == []

        checkpoint_module.save_checkpoint(
            arch="mlp",
            model=model,
            model_path=model_path,
            epoch_number=2,
            dropout=0.0,
            normalize=True,
            action_space="joint_delta",
            norm_stats=norm_stats,
        )
        saved = torch.load(model_path, map_location="cpu", weights_only=False)
        assert saved["epoch"] == 2
        assert list(model_path.parent.glob("*.tmp")) == []

    print("Atomic checkpoint smoke test passed")


if __name__ == "__main__":
    main()
