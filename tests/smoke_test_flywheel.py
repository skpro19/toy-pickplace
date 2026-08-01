from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import threading

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from flywheel import (  # noqa: E402
    append_round_metrics,
    expert_npz_dir_for_run,
    expert_save_images_for_arch,
    make_dagger_round_seeds,
    make_flywheel_seeds,
    make_sample_ratios,
    next_flywheel_run_name,
    parse_args,
    select_best_round,
    validate_expert_npz_for_arch,
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

    flywheel_seeds = make_flywheel_seeds(global_seed=0)
    assert flywheel_seeds == make_flywheel_seeds(global_seed=0)
    assert len(flywheel_seeds) == 4
    assert len(set(flywheel_seeds.values())) == 4
    assert flywheel_seeds != make_flywheel_seeds(global_seed=1)

    assert np.allclose(
        make_sample_ratios(
            expert_ratio=0.5,
            dagger_source_count=4,
            dagger_recency_decay=1.0,
        ),
        [0.5, 0.125, 0.125, 0.125, 0.125],
    )
    recency_ratios = make_sample_ratios(
        expert_ratio=0.5,
        dagger_source_count=2,
        dagger_recency_decay=0.8,
    )
    assert np.allclose(recency_ratios, [0.5, 2.0 / 9.0, 2.5 / 9.0])
    assert np.isclose(sum(recency_ratios), 1.0)
    assert recency_ratios[-1] > recency_ratios[-2]
    for invalid_decay in (0.0, -0.1, 1.1):
        try:
            make_sample_ratios(
                expert_ratio=0.5,
                dagger_source_count=2,
                dagger_recency_decay=invalid_decay,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid DAgger recency decay was accepted: {invalid_decay}"
            )

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
            "dropout: 0.2\n"
            "dagger_intervention_ratio: 0.7\n"
            "dagger_recency_decay: 0.8\n"
            "dataloader_workers: 4\n"
            "persistent_workers: true\n"
            "arch: mlp\n"
            "train_capture_hz: 60\n"
            "eval_capture_hz: 60\n"
            "final_eval_episodes: 200\n"
            "final_eval_seed: 17\n"
            "final_eval_workers: 8\n"
            "final_eval_capture_hz: 30\n"
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
        assert args.dropout == 0.2
        assert args.dagger_intervention_ratio == 0.7
        assert args.dagger_recency_decay == 0.8
        assert args.dataloader_workers == 4
        assert args.persistent_workers
        assert args.global_seed == 0
        assert args.arch == "mlp"
        assert args.final_eval_episodes == 200
        assert args.final_eval_seed == 17
        assert args.final_eval_workers == 8
        assert args.final_eval_capture_hz == 30

        config_path.write_text(
            "global_seed: 11\n"
            "arch: vision_mlp\n"
            "dropout: 0.1\n"
            "train_capture_hz: 60\n"
            "eval_capture_hz: 60\n"
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
        assert args.global_seed == 11
        assert args.arch == "vision_mlp"
        assert args.dropout == 0.1
        assert args.dagger_recency_decay == 1.0
        assert not args.persistent_workers

        config_path.write_text(
            "arch: vision_mlp\n"
            "dropout: 0.1\n"
            "train_capture_hz: 60\n"
            "eval_capture_hz: 60\n"
            "dataloader_workers: 0\n"
            "persistent_workers: true\n"
        )
        try:
            sys.argv = [
                "flywheel.py",
                "--config",
                str(config_path),
            ]
            try:
                parse_args()
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError("persistent workers with zero workers were accepted")
        finally:
            sys.argv = original_argv

        expert_dir = expert_npz_dir_for_run(arch="vision_mlp", run_name="run-test")
        assert expert_dir == Path("data/flywheel/vision_mlp/run-test/expert")

        assert not expert_save_images_for_arch(arch="mlp")
        assert expert_save_images_for_arch(arch="vision_mlp")

        vision_expert_dir = root / "vision-expert"
        vision_expert_dir.mkdir()
        np.savez_compressed(
            vision_expert_dir / "episode_0.npz",
            obs=np.zeros((2, 45), dtype=np.float32),
            actions=np.zeros((2, 8), dtype=np.float32),
        )
        try:
            validate_expert_npz_for_arch(
                expert_npz_dir=vision_expert_dir,
                arch="vision_mlp",
            )
        except ValueError as error:
            assert "img_obs" in str(error)
        else:
            raise AssertionError("vision expert npz without img_obs was accepted")

        validate_expert_npz_for_arch(
            expert_npz_dir=vision_expert_dir,
            arch="mlp",
        )

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
            "config": {
                "final_eval_episodes": 200,
                "final_eval_seed": 17,
                "final_eval_workers": 8,
                "final_eval_capture_hz": 30,
            },
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
        saved_metrics = json.loads(metrics_path.read_text())
        assert saved_metrics["config"]["final_eval_episodes"] == 200
        assert saved_metrics["config"]["final_eval_seed"] == 17
        assert saved_metrics["config"]["final_eval_workers"] == 8
        assert saved_metrics["config"]["final_eval_capture_hz"] == 30

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
