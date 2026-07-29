from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / ".opencode/commands/scripts/flywheel-4090"


def main() -> None:
    workflow = (PROJECT_ROOT / ".opencode/commands/flywheel-4090.md").read_text()
    runner = (SCRIPT_ROOT / "runner.sh").read_text()
    backup = (SCRIPT_ROOT / "ckpt-bkp-wrapper.sh").read_text()
    supervisor = (SCRIPT_ROOT / "supervisor.sh").read_text()
    heldout = (SCRIPT_ROOT / "heldout-eval.sh").read_text()
    destroy = (SCRIPT_ROOT / "destroy-instance.sh").read_text()

    request_read = 'cat "${_C}/state/backup-final-requested"'
    backup_command = "python scripts/s3_backup.py upload"
    acknowledgement = 'write_marker "${_C}/state/backup-final-succeeded"'
    assert request_read in backup
    assert acknowledgement in backup
    assert backup.index(request_read) < backup.index(backup_command)
    assert backup.index(backup_command) < backup.index(acknowledgement)

    assert 'final_backup_succeeded" = "$FINAL_BACKUP_TOKEN' in supervisor
    assert "succeeded_stamp" not in supervisor
    assert "launching held-out eval from local data" not in supervisor
    assert 'cleanup "final-backup-failed"' in supervisor

    assert 'if [ "$#" -ne 8 ]' in supervisor
    assert 'write_status "supervisor-ready"' in supervisor
    assert 'write_status "supervisor-running"' in supervisor
    assert 'tmux has-session -t flywheel-run' in supervisor
    assert 'cleanup "runner-disappeared"' in supervisor
    assert 'tmux has-session -t heldout-eval' in supervisor
    assert 'cleanup "heldout-disappeared"' in supervisor
    assert 'cleanup "heldout-startup-failed"' in supervisor
    assert "FINAL_BACKUP_DEADLINE=$(( $(date +%s) + 10800 ))" in supervisor
    assert "on_supervisor_exit" in supervisor
    assert "emergency_validation_cleanup" in supervisor
    assert '"${SCRIPT_DIR}/destroy-instance.sh" "$INSTANCE_ID"' in supervisor

    assert 'write_marker "${_C}/state/runner-pid"' in runner
    assert runner.index("setsid bash") < runner.index(
        'write_marker "${_C}/state/run-status" "running"'
    )
    assert 'trap on_exit EXIT' in runner

    assert 'write_marker "${_C}/state/backup-running"' in backup
    assert 'write_marker "${_C}/state/backup-artifact-ready"' in backup
    assert "timeout --signal=TERM --kill-after=30s 30m" in backup
    assert "results/flywheel/${_A}/${_R}" not in backup.split(
        'write_marker "${_C}/state/backup-artifact-ready"'
    )[0]

    assert 'write_marker "${_C}/state/heldout-running"' in heldout
    assert 'trap on_exit EXIT' in heldout

    assert "printf -v SUPERVISOR_COMMAND '%q '" in workflow
    assert "Require one complete 30-second monitor cycle" in workflow
    assert 'test "$supervisor_outcome" = "supervisor-running"' in workflow
    assert "state/launch-authorized" in workflow
    assert "RUN_COLLISION_OUTPUT=" in workflow
    assert "COLLISIONS_OK:" in workflow
    assert "destroy-instance.sh \"$INSTANCE_ID\"" in workflow

    assert "while true; do" in destroy
    assert "vastai show instances --raw" in destroy
    assert 'if [ "$count" = "0" ]' in destroy
    assert 'echo "[]"' not in destroy
    assert 'candidate_supervisor_session="flywheel-supervisor-$index"' in workflow
    assert 'test -e "/tmp/toy-pickplace-flywheel-$index.owner"' in workflow

    expected_objects = (
        "metrics.json",
        "final_scores.json",
        "final-placement-score.png",
        "final-score-curve.png",
    )
    for filename in expected_objects:
        assert filename in heldout
        assert filename in supervisor
    assert "final_scores_comparison.png" not in heldout
    assert "final_scores_comparison.png" not in supervisor
    assert "remote sha256 mismatch" in heldout

    print("Flywheel 4090 workflow smoke test passed.")


if __name__ == "__main__":
    main()
