#!/bin/bash
# Network gate — tests RTT and upload speed to S3.
# Runs on the Vast.ai instance during provisioning, before cloning.
#
# Environment:
#   S3_PRESIGNED_PUT    – presigned S3 PUT URL for test upload
#   S3_PRESIGNED_DELETE – presigned S3 DELETE URL for cleanup
#
# Exits 0 (accept) or 1 (reject).

set -o pipefail

RTT_MAX_MS=${NETWORK_GATE_RTT_MS:-500}
UPLOAD_MIN_KBPS=${NETWORK_GATE_UPLOAD_KBPS:-1000}

TEST_FILE="/tmp/.netgate-test.bin"
dd if=/dev/zero bs=1M count=1 of="$TEST_FILE" 2>/dev/null

# ---- RTT test ----
# 2>/dev/null avoids Vast.ai SSH banner on stderr
echo "=== RTT to S3 ap-south-1 ==="
RTT_TOTAL=$(curl -s -o /dev/null -w "%{time_total}" --max-time 10 \
  "https://s3.ap-south-1.amazonaws.com" 2>/dev/null)
RTT_MS=$(awk "BEGIN { printf \"%.0f\", ${RTT_TOTAL:-9.999} * 1000 }" 2>/dev/null)
RTT_MS=${RTT_MS:-9999}
echo "RTT: ${RTT_TOTAL}s / ${RTT_MS} ms  (max: ${RTT_MAX_MS} ms)"

if [ "$RTT_MS" -gt "$RTT_MAX_MS" ]; then
  echo ""
  echo "REJECTED: RTT ${RTT_MS} ms exceeds max ${RTT_MAX_MS} ms"
  curl -s -o /dev/null -X DELETE "${S3_PRESIGNED_DELETE}" 2>/dev/null || true
  rm -f "$TEST_FILE"
  exit 1
fi

# ---- Upload speed test ----
echo ""
echo "=== Upload speed to S3 ==="
UPLOAD_BPS=$(curl -s -o /dev/null -w "%{speed_upload}" --max-time 30 \
  -T "$TEST_FILE" "${S3_PRESIGNED_PUT}" 2>/dev/null)
UPLOAD_KBPS=$(awk "BEGIN { printf \"%.0f\", ${UPLOAD_BPS:-0} / 1024 }" 2>/dev/null)
UPLOAD_KBPS=${UPLOAD_KBPS:-0}
echo "Upload speed: ${UPLOAD_KBPS} KB/s  (min: ${UPLOAD_MIN_KBPS} KB/s)"

if [ "$UPLOAD_KBPS" -lt "$UPLOAD_MIN_KBPS" ]; then
  echo ""
  echo "REJECTED: Upload ${UPLOAD_KBPS} KB/s below min ${UPLOAD_MIN_KBPS} KB/s"
  curl -s -o /dev/null -X DELETE "${S3_PRESIGNED_DELETE}" 2>/dev/null || true
  rm -f "$TEST_FILE"
  exit 1
fi

# ---- Cleanup ----
echo ""
curl -s -o /dev/null -X DELETE "${S3_PRESIGNED_DELETE}" 2>/dev/null || true
rm -f "$TEST_FILE"
echo "PASSED  RTT=${RTT_MS}ms  upload=${UPLOAD_KBPS} KB/s"
exit 0
