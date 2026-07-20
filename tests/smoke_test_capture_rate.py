from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from data import DataCollector  # noqa: E402
from sim import SimEnv  # noqa: E402


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
    print("60 Hz capture smoke test passed.")


if __name__ == "__main__":
    main()
