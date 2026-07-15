import json
from pathlib import Path
import sys
import tempfile

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from dagger_metrics import (  # noqa: E402
    plot_dagger_round,
    plot_dagger_round_trends,
    summarize_dagger_round,
    write_dagger_metrics,
)


def save_episode(
    *,
    path: Path,
    arm_disagreement: list[float],
    execute_expert: list[bool],
    gripper_disagreement: list[bool],
    placement_success: bool,
) -> None:
    np.savez_compressed(
        path,
        arm_disagreement=np.asarray(arm_disagreement, dtype=np.float32),
        execute_expert=np.asarray(execute_expert, dtype=np.bool_),
        gripper_disagreement=np.asarray(gripper_disagreement, dtype=np.bool_),
        intervention_mode=np.asarray("threshold"),
        intervention_threshold=np.asarray(0.25, dtype=np.float32),
        intervention_steps=np.asarray(2, dtype=np.int32),
        grasped=np.asarray(True),
        lifted=np.asarray(True),
        tray_reached=np.asarray(placement_success),
        lowered_to_tray=np.asarray(placement_success),
        released_over_tray=np.asarray(placement_success),
        placement_success=np.asarray(placement_success),
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        dagger_dir = root / "dagger"
        dagger_dir.mkdir()
        save_episode(
            path=dagger_dir / "pick_place_000000.npz",
            arm_disagreement=[0.1, 0.3, 0.4],
            execute_expert=[False, True, True],
            gripper_disagreement=[False, False, True],
            placement_success=True,
        )
        save_episode(
            path=dagger_dir / "pick_place_000001.npz",
            arm_disagreement=[0.2, 0.6],
            execute_expert=[True, False],
            gripper_disagreement=[False, True],
            placement_success=False,
        )

        summary, episodes = summarize_dagger_round(
            dagger_dir=dagger_dir,
            round_index=1,
        )
        assert summary["episodes"] == 2
        assert summary["steps"] == 5
        assert np.isclose(summary["expert_action_fraction"], 0.6)
        assert np.isclose(summary["threshold_exceedance_fraction"], 0.6)
        assert np.isclose(summary["gripper_disagreement_fraction"], 0.4)
        assert summary["expert_control_segments"] == 2
        assert np.isclose(summary["expert_segment_steps"]["mean"], 1.5)
        assert summary["expert_segment_steps"]["max"] == 2
        assert np.isclose(summary["task_success_rates"]["placement_success"], 0.5)

        metrics_path = dagger_dir / "metrics.json"
        round_plot_path = root / "plots" / "round-001-dagger.png"
        trends_plot_path = root / "plots" / "dagger-round-trends.png"
        write_dagger_metrics(metrics_path=metrics_path, metrics=summary)
        plot_dagger_round(
            episodes=episodes,
            summary=summary,
            save_path=round_plot_path,
        )
        plot_dagger_round_trends(
            round_summaries=[summary],
            save_path=trends_plot_path,
        )

        assert json.loads(metrics_path.read_text())["steps"] == 5
        assert round_plot_path.is_file()
        assert trends_plot_path.is_file()

    print("DAgger metrics smoke test passed.")


if __name__ == "__main__":
    main()
