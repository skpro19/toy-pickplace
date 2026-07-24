import re
from pathlib import Path


def next_run_name(
    *,
    base_name: str,
    checkpoint_root: Path,
    log_root: Path,
) -> str:
    existing_indices = []
    pattern = re.compile(rf"^(\d+)_({re.escape(base_name)})$")

    for root in (checkpoint_root, log_root):
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match is not None:
                existing_indices.append(int(match.group(1)))

    next_idx = max(existing_indices, default=0) + 1
    return f"{next_idx:03d}_{base_name}"


def make_run_dirs(
    *,
    base_name: str,
    npz_folders: list[Path],
    num_epochs: int,
    checkpoint_root: Path,
    log_root: Path,
) -> tuple[str, Path, Path]:
    n_episodes = sum(len(list(Path(data_dir).glob("*.npz"))) for data_dir in npz_folders)
    base_name = f"{base_name}_eps{n_episodes}_epochs{num_epochs}"
    while True:
        run_name = next_run_name(
            base_name=base_name,
            checkpoint_root=checkpoint_root,
            log_root=log_root,
        )
        checkpoint_dir = checkpoint_root / run_name
        log_dir = log_root / run_name
        if checkpoint_dir.exists() or log_dir.exists():
            continue

        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        log_dir.mkdir(parents=True, exist_ok=False)
        return run_name, checkpoint_dir, log_dir
