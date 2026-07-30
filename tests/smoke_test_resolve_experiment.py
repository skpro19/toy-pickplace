import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESOLVER = REPO_ROOT / "scripts" / "resolve_experiment.py"


def _run_resolver(*, experiment: Path, seed: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RESOLVER), "--experiment", str(experiment), "--seed", str(seed)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def _run_resolver_sha256(*, experiment: Path, seed: int) -> str:
    result = subprocess.run(
        [sys.executable, str(RESOLVER), "--experiment", str(experiment), "--seed", str(seed), "--sha256"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return result.stdout.strip()


def _run_resolver_manifest(*, experiment: Path, seed: int) -> dict:
    result = subprocess.run(
        [sys.executable, str(RESOLVER), "--experiment", str(experiment), "--seed", str(seed), "--json-manifest"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return json.loads(result.stdout)


def _write_temp_experiment(*, content: str, parent_dir: Path) -> Path:
    exp_dir = parent_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_path = exp_dir / "test.yaml"
    exp_path.write_text(content)
    return exp_path


def _write_temp_base(*, content: str, suite_dir: Path) -> Path:
    base_path = suite_dir.parent / f"{suite_dir.name}.yaml"
    base_path.write_text(content)
    return base_path


def main() -> None:
    suite = tempfile.TemporaryDirectory(dir=REPO_ROOT / "configs" / "flywheel" / "mlp_vision", prefix="test-suite-")
    suite_dir = Path(suite.name)
    suite_name = suite_dir.name
    experiments_dir = suite_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    base_yaml = textwrap.dedent("""\
        run_name: null
        arch: vision_mlp
        dagger_rounds: 10
        num_expert_episodes: 50
        max_steps: 8000
        train_capture_hz: 60
        epochs: 100
        batch_size: 256
        early_stop_patience: 50
        dataloader_workers: 4
        persistent_workers: true
        expert_ratio: 0.5
        dagger_intervention_ratio: 0.8
        dagger_recency_decay: 1.0
        intervention_threshold: 0.1
        intervention_steps: 50
        dagger_episodes: 25
        rollout_max_steps: 1400
        eval_interval: 20
        eval_episodes: 10
        eval_max_steps: 1400
        eval_capture_hz: 60
        mode: mode-b
        final_eval_episodes: 50
        final_eval_seed: 20260716
        final_eval_workers: 6
        final_eval_capture_hz: 60
        workers: 6
        global_seed: 0
    """)
    # Write base at mlp_vision/<suite_name>.yaml
    base_path = suite_dir.parent / f"{suite_name}.yaml"
    base_path.write_text(base_yaml)

    base_rel_from_experiments = f"../../{suite_name}.yaml"

    # --- baseline experiment ---
    baseline_content = textwrap.dedent(f"""\
        description: test baseline
        base_config: {base_rel_from_experiments}
        overrides: {{}}
        global_seeds:
          - 0
          - 42
          - 99
    """)
    baseline_path = experiments_dir / "baseline.yaml"
    baseline_path.write_text(baseline_content)

    # --- override experiment ---
    override_content = textwrap.dedent(f"""\
        description: test override
        base_config: {base_rel_from_experiments}
        overrides:
          num_expert_episodes: 200
          batch_size: 512
          dagger_recency_decay: 0.8
        global_seeds:
          - 0
          - 42
    """)
    override_path = experiments_dir / "override.yaml"
    override_path.write_text(override_content)

    print("--- test: baseline resolves correctly ---")
    result = _run_resolver(experiment=baseline_path, seed=0)
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(result.stdout)
    assert config["global_seed"] == 0
    assert config["num_expert_episodes"] == 50
    assert config["arch"] == "vision_mlp"
    print("  PASS")

    print("--- test: seed replaces global_seed ---")
    result = _run_resolver(experiment=baseline_path, seed=42)
    config = yaml.safe_load(result.stdout)
    assert config["global_seed"] == 42
    assert result.returncode == 0
    print("  PASS")

    print("--- test: overrides apply correctly ---")
    result = _run_resolver(experiment=override_path, seed=0)
    config = yaml.safe_load(result.stdout)
    assert config["num_expert_episodes"] == 200
    assert config["batch_size"] == 512
    assert config["dagger_recency_decay"] == 0.8
    assert config["global_seed"] == 0
    assert config["dagger_rounds"] == 10  # unchanged from base
    print("  PASS")

    print("--- test: SHA-256 is deterministic ---")
    sha1 = _run_resolver_sha256(experiment=baseline_path, seed=0)
    sha2 = _run_resolver_sha256(experiment=baseline_path, seed=0)
    assert sha1 == sha2
    assert len(sha1) == 64
    print("  PASS")

    print("--- test: different seeds produce different SHA-256 ---")
    sha_0 = _run_resolver_sha256(experiment=baseline_path, seed=0)
    sha_42 = _run_resolver_sha256(experiment=baseline_path, seed=42)
    assert sha_0 != sha_42
    print("  PASS")

    print("--- test: different overrides produce different SHA-256 ---")
    sha_baseline = _run_resolver_sha256(experiment=baseline_path, seed=0)
    sha_override = _run_resolver_sha256(experiment=override_path, seed=0)
    assert sha_baseline != sha_override
    print("  PASS")

    print("--- test: invalid seed rejected ---")
    result = _run_resolver(experiment=baseline_path, seed=999)
    assert result.returncode != 0
    assert "not in the experiment" in result.stderr
    print("  PASS")

    print("--- test: JSON manifest includes all fields ---")
    manifest = _run_resolver_manifest(experiment=override_path, seed=42)
    assert manifest["suite"] == suite_name
    assert manifest["experiment_name"] == "override"
    assert manifest["description"] == "test override"
    assert manifest["global_seed"] == 42
    assert "resolved_config_sha256" in manifest
    assert manifest["experiment_config"].startswith("configs/")
    assert manifest["base_config"].startswith("configs/")
    print("  PASS")

    # --- schema validation: global_seed in overrides ---
    print("--- test: global_seed in overrides rejected ---")
    bad_content = textwrap.dedent(f"""\
        description: bad
        base_config: {base_rel_from_experiments}
        overrides:
          global_seed: 5
        global_seeds:
          - 0
    """)
    bad_path = _write_temp_experiment(content=bad_content, parent_dir=suite_dir)
    result = _run_resolver(experiment=bad_path, seed=0)
    assert result.returncode != 0
    assert "global_seed is not allowed in overrides" in result.stderr
    print("  PASS")

    # --- schema validation: unknown experiment key ---
    print("--- test: unknown experiment key rejected ---")
    bad_content = textwrap.dedent(f"""\
        description: bad
        base_config: {base_rel_from_experiments}
        overrides: {{}}
        global_seeds: [0]
        extra_field: nope
    """)
    bad_path = _write_temp_experiment(content=bad_content, parent_dir=suite_dir)
    result = _run_resolver(experiment=bad_path, seed=0)
    assert result.returncode != 0
    assert "unknown experiment keys" in result.stderr
    print("  PASS")

    # --- schema validation: unknown override key ---
    print("--- test: unknown override key rejected ---")
    bad_content = textwrap.dedent(f"""\
        description: bad
        base_config: {base_rel_from_experiments}
        overrides:
          nonexistent_param: 123
        global_seeds: [0]
    """)
    bad_path = _write_temp_experiment(content=bad_content, parent_dir=suite_dir)
    result = _run_resolver(experiment=bad_path, seed=0)
    assert result.returncode != 0
    assert "unknown override keys" in result.stderr
    print("  PASS")

    # --- schema validation: duplicate seeds ---
    print("--- test: duplicate seeds rejected ---")
    bad_content = textwrap.dedent(f"""\
        description: bad
        base_config: {base_rel_from_experiments}
        overrides: {{}}
        global_seeds: [0, 42, 0]
    """)
    bad_path = _write_temp_experiment(content=bad_content, parent_dir=suite_dir)
    result = _run_resolver(experiment=bad_path, seed=0)
    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()
    print("  PASS")

    # --- schema validation: negative seed ---
    print("--- test: negative seed rejected ---")
    bad_content = textwrap.dedent(f"""\
        description: bad
        base_config: {base_rel_from_experiments}
        overrides: {{}}
        global_seeds: [-1, 0]
    """)
    bad_path = _write_temp_experiment(content=bad_content, parent_dir=suite_dir)
    result = _run_resolver(experiment=bad_path, seed=0)
    assert result.returncode != 0
    assert "non-negative" in result.stderr
    print("  PASS")

    # --- schema validation: empty global_seeds ---
    print("--- test: empty global_seeds rejected ---")
    bad_content = textwrap.dedent(f"""\
        description: bad
        base_config: {base_rel_from_experiments}
        overrides: {{}}
        global_seeds: []
    """)
    bad_path = _write_temp_experiment(content=bad_content, parent_dir=suite_dir)
    result = _run_resolver(experiment=bad_path, seed=0)
    assert result.returncode != 0
    assert "non-empty" in result.stderr
    print("  PASS")

    # --- schema validation: missing description ---
    print("--- test: missing description rejected ---")
    bad_content = textwrap.dedent(f"""\
        base_config: {base_rel_from_experiments}
        overrides: {{}}
        global_seeds: [0]
    """)
    bad_path = _write_temp_experiment(content=bad_content, parent_dir=suite_dir)
    result = _run_resolver(experiment=bad_path, seed=0)
    assert result.returncode != 0
    assert "description" in result.stderr.lower()
    print("  PASS")

    # --- run_name in overrides ---
    print("--- test: run_name in overrides rejected ---")
    bad_content = textwrap.dedent(f"""\
        description: bad
        base_config: {base_rel_from_experiments}
        overrides:
          run_name: override_me
        global_seeds: [0]
    """)
    bad_path = _write_temp_experiment(content=bad_content, parent_dir=suite_dir)
    result = _run_resolver(experiment=bad_path, seed=0)
    assert result.returncode != 0
    assert "run_name" in result.stderr
    print("  PASS")

    suite.cleanup()
    base_path.unlink(missing_ok=True)
    print("\nAll resolve_experiment smoke tests passed.")


if __name__ == "__main__":
    main()
