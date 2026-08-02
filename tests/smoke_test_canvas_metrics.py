import json
from pathlib import Path
import sys
import tempfile

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from get_canvas_metrics import CanvasMetricsError, compute_strategy_metrics  # noqa: E402

CONFIG_SLUG = "BASEv3_num-expert-episodes-200"


def final_scores_data(*, placement: list[float]) -> dict:
    rounds = [
        {
            "round": round_index,
            "final_placement_success_rate": rate,
            "final_metrics": {
                "grasp_rate": rate + 0.2,
                "lift_rate": rate + 0.1,
                "tray_reach_rate": rate,
                "lowered_to_tray_rate": rate,
                "released_over_tray_rate": rate,
                "placement_success_rate": rate,
            },
        }
        for round_index, rate in enumerate(placement)
    ]
    best_round = max(rounds, key=lambda item: item["final_placement_success_rate"])
    return {
        "overall_best_round": best_round["round"],
        "overall_best_placement_success_rate": best_round["final_placement_success_rate"],
        "rounds": rounds,
    }


def metrics_data(*, placement: list[float]) -> dict:
    return {
        "rounds": [
            {"round": round_index, "eval_metrics": {"placement_success_rate": rate}}
            for round_index, rate in enumerate(placement)
        ]
    }


def write_run(
    *,
    root: Path,
    seed: int,
    has_metrics: bool,
    has_final_scores: bool = True,
    held_out_rounds: int = 11,
) -> str:
    run_name = f"2026-08-02_14-04-30_1785659670318738614_{CONFIG_SLUG}_seed{seed}"
    run_dir = root / "results" / "flywheel" / "vision_mlp" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    held_out = [0.30 + 0.05 * round_index for round_index in range(held_out_rounds)]
    final_scores_path = run_dir / "final_scores.json"
    if has_final_scores:
        final_scores_path.write_text(json.dumps(final_scores_data(placement=held_out)))
    elif final_scores_path.is_file():
        final_scores_path.unlink()

    metrics_path = run_dir / "metrics.json"
    if has_metrics:
        train = [0.25 + 0.05 * round_index for round_index in range(held_out_rounds)]
        metrics_path.write_text(json.dumps(metrics_data(placement=train)))
    elif metrics_path.is_file():
        metrics_path.unlink()
    return run_name


def compute(*, root: Path, run_names: list[str], strict: bool = True) -> dict:
    return compute_strategy_metrics(
        run_names=run_names,
        results_root=root / "results" / "flywheel",
        arch="vision_mlp",
        from_s3=False,
        s3_bucket="s3://toy-pickplace",
        aws_profile=None,
        strict=strict,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        complete_runs = [
            write_run(root=root, seed=0, has_metrics=True),
            write_run(root=root, seed=420, has_metrics=True),
        ]
        complete = compute(root=root, run_names=complete_runs, strict=True)
        train_curve = [25.0 + 5.0 * round_index for round_index in range(11)]
        assert complete["train"] == {"seed0": train_curve, "seed420": train_curve}
        assert complete["mean_train"] == train_curve
        assert complete["warnings"] == []

        partial_runs = [
            write_run(root=root, seed=0, has_metrics=True),
            write_run(root=root, seed=420, has_metrics=False),
        ]
        partial = compute(root=root, run_names=partial_runs, strict=False)
        assert partial["train"] == {"seed0": train_curve}
        assert partial["mean_train"] == train_curve
        assert any(
            "seed420" in warning and "metrics.json" in warning
            for warning in partial["warnings"]
        ), "missing metrics.json warning was not surfaced"
        assert any(
            "excluded" in warning and "seed420" in warning and "mean" in warning
            for warning in partial["warnings"]
        ), "excluded seeds were not reported for the train cross-seed mean"

        missing_metrics_runs = [
            write_run(root=root, seed=0, has_metrics=True),
            write_run(root=root, seed=420, has_metrics=False),
        ]
        try:
            compute(root=root, run_names=missing_metrics_runs, strict=True)
        except CanvasMetricsError as error:
            assert "metrics.json" in str(error)
        else:
            raise AssertionError("strict mode should fail when metrics.json is missing")

        missing_final_scores_runs = [
            write_run(root=root, seed=0, has_metrics=True, has_final_scores=True),
            write_run(root=root, seed=420, has_metrics=True, has_final_scores=False),
        ]
        try:
            compute(root=root, run_names=missing_final_scores_runs, strict=True)
        except CanvasMetricsError as error:
            assert "final_scores.json" in str(error)
        else:
            raise AssertionError("strict mode should fail when final_scores.json is missing")

        unequal_round_runs = [
            write_run(root=root, seed=0, has_metrics=True, held_out_rounds=11),
            write_run(root=root, seed=420, has_metrics=True, held_out_rounds=9),
        ]
        try:
            compute(root=root, run_names=unequal_round_runs, strict=True)
        except CanvasMetricsError as error:
            assert "different round counts" in str(error)
        else:
            raise AssertionError("strict mode should fail when seed round counts differ")

    print("Canvas metrics strict-mode smoke test passed.")


if __name__ == "__main__":
    main()
