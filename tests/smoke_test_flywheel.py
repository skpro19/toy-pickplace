from pathlib import Path
import sys
import tempfile

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from flywheel import append_round_metrics, make_dagger_round_seeds  # noqa: E402
from eval import EVAL_METRIC_VERSION  # noqa: E402


def main() -> None:
    seeds = make_dagger_round_seeds(seed=42, rounds=10)

    assert seeds == make_dagger_round_seeds(seed=42, rounds=10)
    assert len(set(seeds)) == len(seeds)
    assert seeds != make_dagger_round_seeds(seed=43, rounds=10)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        metrics_path = root / "metrics.json"
        checkpoint_path = root / "best.pt"
        rounds: list[dict[str, object]] = []
        common_args = {
            "metrics_path": metrics_path,
            "run_name": "test",
            "rounds": rounds,
            "beta": None,
            "dagger_seed": None,
            "data_dirs": [],
            "sample_ratios": [],
            "best_checkpoint": checkpoint_path,
            "config": {},
            "eval_seed": 42,
            "eval_episodes": 1,
            "eval_max_steps": 1,
        }
        torch.save(
            {
                "epoch": 1,
                "eval_score": 0.5,
                "eval_metric_version": EVAL_METRIC_VERSION,
            },
            checkpoint_path,
        )
        append_round_metrics(round_index=0, **common_args)

        torch.save(
            {
                "epoch": 1,
                "eval_score": 0.6,
                "eval_metric_version": EVAL_METRIC_VERSION - 1,
            },
            checkpoint_path,
        )
        try:
            append_round_metrics(round_index=1, **common_args)
        except ValueError:
            pass
        else:
            raise AssertionError("Mixed evaluation metric versions were accepted")

    print("Flywheel DAgger seed smoke test passed.")


if __name__ == "__main__":
    main()
