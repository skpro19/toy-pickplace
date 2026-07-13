from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from flywheel import make_dagger_round_seeds  # noqa: E402


def main() -> None:
    seeds = make_dagger_round_seeds(seed=42, rounds=10)

    assert seeds == make_dagger_round_seeds(seed=42, rounds=10)
    assert len(set(seeds)) == len(seeds)
    assert seeds != make_dagger_round_seeds(seed=43, rounds=10)

    print("Flywheel DAgger seed smoke test passed.")


if __name__ == "__main__":
    main()
