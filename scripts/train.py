from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.eval import (  # noqa: E402
    DEFAULT_EVAL_SELECTION_MODE,
    EVAL_SELECTION_MODES,
    EvalSelectionMode,
)
from scripts.train_core.engine import run_training  # noqa: E402
from scripts.train_core.recipes.registry import get_recipe  # noqa: E402
from scripts.train_core.run_dirs import make_run_dirs  # noqa: E402
from scripts.train_core.types import TrainConfig  # noqa: E402

# Backward-compatible re-exports for callers that import helpers from train.py.
from scripts.train_core.checkpoint import save_checkpoint  # noqa: E402
from scripts.train_core.recipes.mlp import prepare_dataset  # noqa: E402


def train(
    *,
    arch: str = "mlp",
    num_epochs: int = 10,
    batch_size: int = 200,
    npz_folders: list[Path],
    checkpoint_dir: Path,
    log_dir: Path,
    eval_capture_hz: float,
    normalize: bool = True,
    action_space: str = "joint_delta",
    sample_ratios: list[float] | None = None,
    dagger_intervention_ratio: float | None = None,
    sample_seed: int = 0,
    eval_interval: int = 1,
    eval_seed: int = 0,
    eval_episodes: int = 100,
    eval_max_steps: int = 1400,
    eval_workers: int = 1,
    dataloader_workers: int = 0,
    persistent_workers: bool = False,
    early_stop_patience: int = 50,
    eval_selection_mode: EvalSelectionMode = DEFAULT_EVAL_SELECTION_MODE,
    init_checkpoint: Path | None = None,
    fresh_optimizer_state: bool = True,
) -> Path:
    config: TrainConfig = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "npz_folders": npz_folders,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
        "normalize": normalize,
        "action_space": action_space,
        "sample_ratios": sample_ratios,
        "dagger_intervention_ratio": dagger_intervention_ratio,
        "sample_seed": sample_seed,
        "eval_interval": eval_interval,
        "eval_seed": eval_seed,
        "eval_episodes": eval_episodes,
        "eval_max_steps": eval_max_steps,
        "eval_workers": eval_workers,
        "eval_capture_hz": eval_capture_hz,
        "dataloader_workers": dataloader_workers,
        "persistent_workers": persistent_workers,
        "early_stop_patience": early_stop_patience,
        "eval_selection_mode": eval_selection_mode,
        "init_checkpoint": init_checkpoint,
        "fresh_optimizer_state": fresh_optimizer_state,
    }
    return run_training(config=config, recipe=get_recipe(arch=arch))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a policy from NPZ demonstrations"
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="mlp",
        choices=["mlp", "vision_mlp"],
        help="Policy architecture / training recipe to use",
    )
    parser.add_argument("--base", type=str, default="action-delta")
    parser.add_argument("--checkpoint_root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--log_root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--npz",
        type=Path,
        nargs="+",
        required=True,
        help="npz folder path(s)",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="joint_delta",
        choices=["joint_delta", "absolute"],
    )
    parser.add_argument(
        "--sample-ratios",
        type=float,
        nargs="+",
        default=None,
        help="Optional per --npz directory timestep sampling ratios, e.g. 0.8 0.2",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed used when --sample-ratios is set",
    )
    parser.add_argument(
        "--dagger-intervention-ratio",
        type=float,
        default=None,
        help="Sampling share for execute_expert frames within DAgger directories",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Save and evaluate every N epochs; the final epoch is always evaluated",
    )

    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-max-steps", type=int, default=1400)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument(
        "--eval-capture-hz",
        type=float,
        default=None,
        help="Policy inference rate used during in-loop checkpoint evaluation",
    )
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep DataLoader worker processes alive between epochs",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=50,
        help="Stop after this many epochs without eval improvement; 0 disables",
    )
    parser.add_argument(
        "--eval-selection-mode",
        choices=EVAL_SELECTION_MODES,
        default=DEFAULT_EVAL_SELECTION_MODE,
        help="Select best checkpoints by weighted score or placement rate first",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.eval_interval < 1:
        parser.error("--eval-interval must be at least 1")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be at least 1")
    if args.eval_max_steps < 1:
        parser.error("--eval-max-steps must be at least 1")
    if args.eval_workers < 1:
        parser.error("--eval-workers must be at least 1")
    if args.eval_capture_hz is None:
        parser.error("--eval-capture-hz is required")
    if args.eval_capture_hz <= 0.0:
        parser.error("--eval-capture-hz must be positive")
    if args.dataloader_workers < 0:
        parser.error("--dataloader-workers must be non-negative")
    if args.persistent_workers and args.dataloader_workers == 0:
        parser.error("--persistent-workers requires --dataloader-workers > 0")
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be non-negative")
    if args.sample_ratios is not None and len(args.sample_ratios) != len(args.npz):
        parser.error(
            f"--sample-ratios length ({len(args.sample_ratios)}) "
            f"must match --npz length ({len(args.npz)})"
        )
    if args.dagger_intervention_ratio is not None and not (
        0.0 <= args.dagger_intervention_ratio <= 1.0
    ):
        parser.error("--dagger-intervention-ratio must be between 0 and 1")
    for npz_folder in args.npz:
        if not npz_folder.is_dir():
            parser.error(f"--npz path is not a directory: {npz_folder}")

    return args


def main():
    args = parse_args()

    _run_name, checkpoint_dir, log_dir = make_run_dirs(
        base_name=args.base,
        npz_folders=args.npz,
        num_epochs=args.epochs,
        checkpoint_root=args.checkpoint_root,
        log_root=args.log_root,
    )

    train(
        arch=args.arch,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        npz_folders=args.npz,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        eval_capture_hz=args.eval_capture_hz,
        action_space=args.action_space,
        normalize=args.normalize,
        sample_ratios=args.sample_ratios,
        dagger_intervention_ratio=args.dagger_intervention_ratio,
        sample_seed=args.sample_seed,
        eval_interval=args.eval_interval,
        eval_seed=args.eval_seed,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        eval_workers=args.eval_workers,
        dataloader_workers=args.dataloader_workers,
        persistent_workers=args.persistent_workers,
        early_stop_patience=args.early_stop_patience,
        eval_selection_mode=args.eval_selection_mode,
    )


if __name__ == "__main__":
    main()
