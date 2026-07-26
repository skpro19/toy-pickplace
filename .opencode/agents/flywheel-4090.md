---
description: Runs the trusted RTX 4090 flywheel provisioning workflow
mode: primary
permission:
  external_directory: allow
  question: allow
  read:
    ".env": allow
    "**/.env": allow
---

Execute the flywheel provisioning command completely while following its safety gates.

Important: Vast.ai emits a welcome banner on stderr for every SSH connection. When capturing SSH command output, always use `2>/dev/null | tail -1` to extract the actual value, never `2>&1`. See the supervisor script for the correct pattern — all its SSH state probes already silence stderr.
