"""Train VisionMLP — delegates to scripts/train.py --arch vision_mlp."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    if "--arch" not in sys.argv:
        sys.argv = [sys.argv[0], "--arch", "vision_mlp", *sys.argv[1:]]

    from train import main

    main()
