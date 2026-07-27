#!/bin/bash
set -o pipefail
export MUJOCO_GL=egl
_R=__RUN_NAME__
_C=__CONTROL_DIR__

fail_heldout() {
  local exit_code="${1:-1}"
  echo "failed ${exit_code}" > "${_C}/state/heldout-failed.tmp"
  mv "${_C}/state/heldout-failed.tmp" "${_C}/state/heldout-failed"
  exit "$exit_code"
}

if test -f "${_C}/state/heldout-completed" || test -f "${_C}/state/heldout-failed"; then
  echo "ERROR: held-out evaluation already ran" >&2; exit 1
fi

METRICS="/workspace/toy-pickplace/results/flywheel/${_R}/metrics.json"
if test ! -f "$METRICS"; then
  echo "ERROR: metrics.json not found at $METRICS" >&2
  fail_heldout 1
fi

cd /workspace/toy-pickplace

/root/.local/bin/uv run --env-file "${_C}/s3-env.env" \
  python scripts/final_score.py --run-name "${_R}" 2>&1 | \
  tee -a "${_C}/logs/heldout-eval-${_R}.log"
exit_code=${PIPESTATUS[0]}

if [ "$exit_code" -ne 0 ]; then
  fail_heldout "$exit_code"
fi

# Validate outputs
for f in "results/flywheel/${_R}/final_scores.json" \
         "results/flywheel/${_R}/final-placement-score.png" \
         "results/flywheel/${_R}/final-score-curve.png"; do
  if test ! -f "/workspace/toy-pickplace/$f"; then
    echo "ERROR: missing output $f" >&2
    fail_heldout 1
  fi
done

# Upload finalized metrics and held-out outputs with content hashes.
upload_and_verify() {
  local src="$1" key="$2"
  /root/.local/bin/uv run --env-file "${_C}/s3-env.env" python -c "
import boto3, hashlib, os, sys
from pathlib import Path
cfg = {}
with open('${_C}/s3-env.env') as f:
    for line in f:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            cfg[k] = v
c = boto3.client('s3', region_name=cfg.get('AWS_REGION', ''))
digest = hashlib.sha256(Path('$src').read_bytes()).hexdigest()
c.upload_file(
    '$src',
    cfg['S3_BUCKET'],
    '$key',
    ExtraArgs={'Metadata': {'sha256': digest}},
)
r = c.head_object(Bucket=cfg['S3_BUCKET'], Key='$key')
sz = r['ContentLength']
if sz != os.path.getsize('$src'):
    print(f'size mismatch: sz vs {os.path.getsize(\"$src\")}', file=sys.stderr)
    sys.exit(1)
if r.get('Metadata', {}).get('sha256') != digest:
    print('sha256 metadata mismatch', file=sys.stderr)
    sys.exit(1)
remote = c.get_object(Bucket=cfg['S3_BUCKET'], Key='$key')['Body'].read()
if hashlib.sha256(remote).hexdigest() != digest:
    print('remote sha256 mismatch', file=sys.stderr)
    sys.exit(1)
print(f'Verified: s3://{cfg[\"S3_BUCKET\"]}/\$key ({sz} bytes)')
" || { echo "upload failed for $2" >&2; return 1; }
}

. "${_C}/s3-env.env"

upload_and_verify \
  "$METRICS" \
  "results/flywheel/${_R}/metrics.json" || fail_heldout 1

upload_and_verify \
  "/workspace/toy-pickplace/results/flywheel/${_R}/final_scores.json" \
  "results/flywheel/${_R}/final_scores.json" || fail_heldout 1

upload_and_verify \
  "/workspace/toy-pickplace/results/flywheel/${_R}/final-placement-score.png" \
  "results/flywheel/${_R}/final-placement-score.png" || fail_heldout 1

upload_and_verify \
  "/workspace/toy-pickplace/results/flywheel/${_R}/final-score-curve.png" \
  "results/flywheel/${_R}/final-score-curve.png" || fail_heldout 1

echo "succeeded 0" > "${_C}/state/heldout-completed.tmp"
mv "${_C}/state/heldout-completed.tmp" "${_C}/state/heldout-completed"
exit 0
