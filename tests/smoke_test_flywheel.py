from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import threading

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from flywheel import (  # noqa: E402
    append_round_metrics,
    expert_npz_dir_for_run,
    make_dagger_round_seeds,
    next_flywheel_run_name,
    parse_args,
    select_best_round,
)
from data import collect_expert_episodes  # noqa: E402
from eval import (  # noqa: E402
    DEFAULT_EVAL_SELECTION_MODE,
    EVAL_METRIC_VERSION,
    eval_selection_key,
)


def main() -> None:
    seeds = make_dagger_round_seeds(seed=42, rounds=10)

    assert seeds == make_dagger_round_seeds(seed=42, rounds=10)
    assert len(set(seeds)) == len(seeds)
    assert seeds != make_dagger_round_seeds(seed=43, rounds=10)

    assert eval_selection_key(
        selection_mode="mode-a",
        mean_score=0.8,
        placement_success_rate=0.2,
    ) > eval_selection_key(
        selection_mode="mode-a",
        mean_score=0.7,
        placement_success_rate=0.3,
    )
    assert eval_selection_key(
        selection_mode="mode-b",
        mean_score=0.7,
        placement_success_rate=0.3,
    ) > eval_selection_key(
        selection_mode="mode-b",
        mean_score=0.8,
        placement_success_rate=0.2,
    )
    assert eval_selection_key(
        selection_mode="mode-b",
        mean_score=0.8,
        placement_success_rate=0.3,
    ) > eval_selection_key(
        selection_mode="mode-b",
        mean_score=0.7,
        placement_success_rate=0.3,
    )

    candidate_rounds = [
        {
            "best_score": 0.8,
            "best_placement_success_rate": 0.2,
        },
        {
            "best_score": 0.7,
            "best_placement_success_rate": 0.3,
        },
    ]
    assert select_best_round(
        rounds=candidate_rounds,
        selection_mode="mode-a",
    ) is candidate_rounds[0]
    assert select_best_round(
        rounds=candidate_rounds,
        selection_mode="mode-b",
    ) is candidate_rounds[1]

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        data_root = root / "data"
        checkpoint_root = root / "checkpoints"
        results_root = root / "results"
        (checkpoint_root / "run-001").mkdir(parents=True)
        (results_root / "run-003").mkdir(parents=True)

        assert next_flywheel_run_name(
            root=data_root,
            occupied_roots=[checkpoint_root, results_root],
        ) == "run-004"
        assert (data_root / "run-004").is_dir()
        assert next_flywheel_run_name(
            root=data_root,
            occupied_roots=[checkpoint_root, results_root],
        ) == "run-005"

        barrier = threading.Barrier(2)

        def reserve_concurrently() -> str:
            barrier.wait()
            return next_flywheel_run_name(
                root=data_root,
                occupied_roots=[checkpoint_root, results_root],
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            reserved_names = list(
                executor.map(lambda _: reserve_concurrently(), range(2))
            )
        assert sorted(reserved_names) == ["run-006", "run-007"]

        metrics_path = root / "metrics.json"
        checkpoint_path = root / "best.pt"

        config_path = root / "flywheel.yaml"
        config_path.write_text(
            "num_expert_episodes: 42\n"
            "max_steps: 5000\n"
            "epochs: 7\n"
            "dagger_intervention_ratio: 0.7\n"
        )
        original_argv = sys.argv
        try:
            sys.argv = [
                "flywheel.py",
                "--config",
                str(config_path),
                "--epochs",
                "9",
            ]
            args = parse_args()
        finally:
            sys.argv = original_argv
        assert args.num_expert_episodes == 42
        assert args.max_steps == 5000
        assert args.epochs == 9
        assert args.dagger_intervention_ratio == 0.7
        assert args.expert_seed == 0

        config_path.write_text(
            "train_seed: 11\n"
            "expert_seed: 22\n"
        )
        try:
            sys.argv = [
                "flywheel.py",
                "--config",
                str(config_path),
            ]
            args = parse_args()
        finally:
            sys.argv = original_argv
        assert args.train_seed == 11
        assert args.expert_seed == 22

        config_path.write_text("train_seed: 11\n")
        try:
            sys.argv = [
                "flywheel.py",
                "--config",
                str(config_path),
            ]
            args = parse_args()
        finally:
            sys.argv = original_argv
        assert args.train_seed == 11
        assert args.expert_seed == 11

        expert_dir = expert_npz_dir_for_run(run_name="run-test")
        assert expert_dir == Path("data/flywheel/run-test/expert")
        collect_expert_episodes(
            episodes=1,
            out_dir=root / "expert",
            seed=0,
            max_steps=5,
        )
        expert_files = sorted((root / "expert").glob("*.npz"))
        assert len(expert_files) == 1
        collect_expert_episodes(
            episodes=1,
            out_dir=root / "expert",
            seed=0,
            max_steps=5,
        )
        assert len(list((root / "expert").glob("*.npz"))) == 1

        rounds: list[dict[str, object]] = []
        common_args = {
            "metrics_path": metrics_path,
            "run_name": "test",
            "rounds": rounds,
            "intervention_threshold": None,
            "dagger_seed": None,
            "dagger_metrics": None,
            "data_dirs": [],
            "sample_ratios": [],
            "best_checkpoint": checkpoint_path,
            "config": {},
            "eval_seed": 42,
            "eval_episodes": 1,
            "eval_max_steps": 1,
            "selection_mode": DEFAULT_EVAL_SELECTION_MODE,
        }
        torch.save(
            {
                "epoch": 1,
                "eval_score": 0.5,
                "eval_metric_version": EVAL_METRIC_VERSION,
                "eval_selection_mode": DEFAULT_EVAL_SELECTION_MODE,
                "eval_metrics": {"placement_success_rate": 0.25},
            },
            checkpoint_path,
        )
        append_round_metrics(round_index=0, **common_args)

        torch.save(
            {
                "epoch": 1,
                "eval_score": 0.6,
                "eval_metric_version": EVAL_METRIC_VERSION - 1,
                "eval_selection_mode": DEFAULT_EVAL_SELECTION_MODE,
                "eval_metrics": {"placement_success_rate": 0.5},
            },
            checkpoint_path,
        )
        try:
            append_round_metrics(round_index=1, **common_args)
        except ValueError:
            pass
        else:
            raise AssertionError("Mixed evaluation metric versions were accepted")

        torch.save(
            {
                "epoch": 1,
                "eval_score": 0.6,
                "eval_metric_version": EVAL_METRIC_VERSION,
                "eval_selection_mode": "mode-a",
                "eval_metrics": {"placement_success_rate": 0.5},
            },
            checkpoint_path,
        )
        try:
            append_round_metrics(round_index=1, **common_args)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched selection mode was accepted")

    print("Flywheel DAgger seed smoke test passed.")


if __name__ == "__main__":
    main()
