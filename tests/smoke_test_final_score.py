from pathlib import Path
import sys
import tempfile


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from final_score import plot_comparison, resolve_final_eval_settings  # noqa: E402


def main() -> None:
    config = {
        "final_eval_episodes": 200,
        "final_eval_seed": 17,
        "final_eval_workers": 8,
        "final_eval_capture_hz": 30,
        "workers": 12,
        "eval_capture_hz": 60,
    }
    settings = resolve_final_eval_settings(
        config=config,
        eval_episodes=None,
        final_eval_seed=None,
        workers=None,
        capture_hz=None,
    )
    assert settings == (200, 17, 8, 30.0)

    overrides = resolve_final_eval_settings(
        config=config,
        eval_episodes=50,
        final_eval_seed=19,
        workers=4,
        capture_hz=20,
    )
    assert overrides == (50, 19, 4, 20)

    for invalid_episodes in (4, 101):
        try:
            resolve_final_eval_settings(
                config=config,
                eval_episodes=invalid_episodes,
                final_eval_seed=None,
                workers=None,
                capture_hz=None,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid episode count accepted: {invalid_episodes}")

    invalid_settings = (
        {"final_eval_seed": -1},
        {"workers": 0},
        {"capture_hz": 0},
    )
    for invalid in invalid_settings:
        try:
            resolve_final_eval_settings(
                config=config,
                eval_episodes=None,
                final_eval_seed=invalid.get("final_eval_seed"),
                workers=invalid.get("workers"),
                capture_hz=invalid.get("capture_hz"),
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid final eval settings accepted: {invalid}")

    with tempfile.TemporaryDirectory() as temp_dir:
        run_results_dir = Path(temp_dir) / "results/flywheel/test-run"
        run_results_dir.mkdir(parents=True)
        plot_comparison(
            final_rounds=[
                {
                    "round": 0,
                    "original_score": 0.5,
                    "final_score": 0.75,
                    "original_placement_success_rate": 0.4,
                    "final_placement_success_rate": 0.7,
                }
            ],
            run_name="test-run",
            save_dir=run_results_dir,
            eval_episodes=100,
            original_episodes=25,
        )
        assert (run_results_dir / "final-placement-score.png").is_file()
        assert (run_results_dir / "final-score-curve.png").is_file()

    print("Final score config smoke test passed.")


if __name__ == "__main__":
    main()
