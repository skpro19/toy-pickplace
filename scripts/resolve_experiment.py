"""Resolve an experiment definition against its base config with a selected seed.

Usage:
    uv run python scripts/resolve_experiment.py \
        --experiment configs/flywheel/mlp_vision/BASE/experiments/baseline.yaml \
        --seed 420

Outputs the resolved flat config YAML to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

_VALID_FLYWHEEL_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "run_name",
        "arch",
        "dagger_rounds",
        "num_expert_episodes",
        "max_steps",
        "train_capture_hz",
        "epochs",
        "batch_size",
        "early_stop_patience",
        "dataloader_workers",
        "persistent_workers",
        "expert_ratio",
        "dagger_intervention_ratio",
        "intervention_threshold",
        "intervention_steps",
        "dagger_episodes",
        "rollout_max_steps",
        "eval_interval",
        "eval_episodes",
        "eval_max_steps",
        "eval_capture_hz",
        "mode",
        "final_eval_episodes",
        "final_eval_seed",
        "final_eval_workers",
        "final_eval_capture_hz",
        "workers",
        "global_seed",
        "warm_start",
        "fresh_optimizer_state",
    }
)

_ALLOWED_EXPERIMENT_KEYS: frozenset[str] = frozenset(
    {"description", "base_config", "overrides", "global_seeds"}
)


def _die(*, message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _validate_experiment_schema(*, experiment: dict[str, object], path: Path) -> None:
    unknown = set(experiment) - _ALLOWED_EXPERIMENT_KEYS
    if unknown:
        _die(message=f"{path}: unknown experiment keys: {sorted(unknown)}")

    if not isinstance(experiment.get("description"), str) or not experiment["description"].strip():
        _die(message=f"{path}: description is required and must be a non-empty string")

    base_config_raw = experiment.get("base_config")
    if not isinstance(base_config_raw, str) or not base_config_raw.strip():
        _die(message=f"{path}: base_config is required and must be a non-empty string")

    overrides = experiment.get("overrides")
    if overrides is not None and not isinstance(overrides, dict):
        _die(message=f"{path}: overrides must be a mapping")

    seeds = experiment.get("global_seeds")
    if not isinstance(seeds, list) or len(seeds) == 0:
        _die(message=f"{path}: global_seeds must be a non-empty list")
    seen: set[int] = set()
    for s in seeds:
        if not isinstance(s, int):
            _die(message=f"{path}: global_seeds must contain only integers, got {type(s).__name__}")
        if s < 0:
            _die(message=f"{path}: global_seed values must be non-negative, got {s}")
        if s in seen:
            _die(message=f"{path}: global_seeds contains duplicate value {s}")
        seen.add(s)


def _validate_overrides(*, overrides: dict[str, object], path: Path) -> None:
    if "global_seed" in overrides:
        _die(message=f"{path}: global_seed is not allowed in overrides; use global_seeds at experiment level")

    unknown = set(overrides) - _VALID_FLYWHEEL_CONFIG_KEYS
    if unknown:
        _die(message=f"{path}: unknown override keys: {sorted(unknown)}")

    if "run_name" in overrides:
        _die(message=f"{path}: run_name is not allowed in overrides")


def _validate_seed_in_definition(*, seed: int, experiment: dict[str, object], path: Path) -> None:
    seeds = experiment["global_seeds"]
    assert isinstance(seeds, list)
    if seed not in seeds:
        _die(message=f"{path}: seed {seed} is not in the experiment's global_seeds list: {seeds}")


def resolve_experiment(
    *,
    experiment_path: Path,
    seed: int,
    repo_root: Path | None = None,
) -> tuple[dict[str, object], str, dict[str, object]]:
    experiment_path = experiment_path.resolve()
    if repo_root is None:
        repo_root = experiment_path.parent
        while repo_root.parent != repo_root and not (repo_root / "pyproject.toml").exists():
            repo_root = repo_root.parent

    if not (repo_root / "pyproject.toml").exists():
        _die(message=f"could not locate repository root; started from {experiment_path}")

    experiment_text = experiment_path.read_text()
    experiment = yaml.safe_load(experiment_text)
    if not isinstance(experiment, dict):
        _die(message=f"{experiment_path}: experiment must be a YAML mapping")

    _validate_experiment_schema(experiment=experiment, path=experiment_path)

    base_config_rel = experiment["base_config"]
    assert isinstance(base_config_rel, str)
    base_config_path = (experiment_path.parent / base_config_rel).resolve()
    if not base_config_path.is_relative_to(repo_root):
        _die(
            message=(
                f"base_config resolves outside the repository: "
                f"{base_config_path} is not under {repo_root}"
            )
        )

    base_text = base_config_path.read_text()
    base = yaml.safe_load(base_text)
    if not isinstance(base, dict):
        _die(message=f"{base_config_path}: base config must be a YAML mapping")

    _validate_seed_in_definition(seed=seed, experiment=experiment, path=experiment_path)

    overrides = experiment.get("overrides")
    if isinstance(overrides, dict):
        _validate_overrides(overrides=overrides, path=experiment_path)

    config: dict[str, object] = {}
    for key, value in base.items():
        if isinstance(key, str):
            config[key] = value
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if isinstance(key, str):
                config[key] = value
    config["global_seed"] = seed

    description = experiment.get("description", "")
    assert isinstance(description, str)

    suite = experiment_path.parent.parent.name

    manifest: dict[str, object] = {
        "experiment_config": str(experiment_path.relative_to(repo_root)),
        "base_config": str(base_config_path.relative_to(repo_root)),
        "suite": suite,
        "experiment_name": experiment_path.stem,
        "description": description,
        "global_seed": seed,
    }

    return config, description, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve an experiment definition against its base config"
    )
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--json-manifest", action="store_true", default=False)
    parser.add_argument("--sha256", action="store_true", default=False)
    args = parser.parse_args()

    config, _description, manifest = resolve_experiment(
        experiment_path=args.experiment,
        seed=args.seed,
    )

    output = yaml.dump(config, default_flow_style=False, sort_keys=False)
    sha256_hex = hashlib.sha256(output.encode()).hexdigest()

    if args.json_manifest:
        manifest["resolved_config_sha256"] = sha256_hex
        json.dump(manifest, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.sha256:
        print(sha256_hex)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
