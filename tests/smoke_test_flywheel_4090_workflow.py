from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / ".opencode/commands/scripts/flywheel-4090"


def main() -> None:
    backup = (SCRIPT_ROOT / "ckpt-bkp-wrapper.sh").read_text()
    supervisor = (SCRIPT_ROOT / "supervisor.sh").read_text()
    heldout = (SCRIPT_ROOT / "heldout-eval.sh").read_text()

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
