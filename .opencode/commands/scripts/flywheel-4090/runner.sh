#!/bin/bash
set -o pipefail
export MUJOCO_GL=egl

_R=__RUN_NAME__
_F=__FLYWHEEL_CONFIG__
_C=__CONTROL_DIR__

echo "running" > "${_C}/state/run-status.tmp"
mv "${_C}/state/run-status.tmp" "${_C}/state/run-status"

cd /workspace/toy-pickplace

/root/.local/bin/uv run python scripts/flywheel.py \
  --config "${_F}" --run-name "${_R}" 2>&1 | \
  tee -a "${_C}/logs/${_R}.log"
exit_code=${PIPESTATUS[0]}

if [ "$exit_code" -eq 0 ]; then
  echo "succeeded 0" > "${_C}/state/completed.tmp"
  mv "${_C}/state/completed.tmp" "${_C}/state/completed"
else
  echo "failed ${exit_code}" > "${_C}/state/failed.tmp"
  mv "${_C}/state/failed.tmp" "${_C}/state/failed"
fi

exit "${exit_code}"
