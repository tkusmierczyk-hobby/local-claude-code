#!/usr/bin/env bash
# auto-tune.sh — wait for a model download to finish, then tune and self-test.
#
# The main model is a ~20 h download on this link, so the tuning sweep and the
# API conformance run happen unattended: start this detached after kicking off a
# pull and read the report in the morning.
#
# Usage: setsid nohup bin/auto-tune.sh [model-file] </dev/null >/dev/null 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODEL="${1:-$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
STATE_DIR="${LOCAL_CLAUDE_STATE:-$HOME/.local-claude}"   # where the launcher keeps it too
REPORT="$STATE_DIR/bench-results.txt"
mkdir -p "$STATE_DIR"

exec >>"$STATE_DIR/auto-tune.log" 2>&1
echo "=== $(date -Is) waiting for $MODEL ==="
while [ ! -f "$MODEL" ]; do sleep 60; done
echo "=== $(date -Is) model present ($(stat -c%s "$MODEL") bytes) ==="

# Tuning restarts the inference server repeatedly, which would kill an active
# Claude Code session. Wait for the machine to be idle first (up to 6 h), then go.
# Only a session on the *local* backend is at risk from a server restart, and
# those set LOCAL_CLAUDE=1. A plain `pgrep -x claude` also matches cloud-backed
# sessions, which made this wait forever with nothing to wait for.
local_session() {
  local pid
  for pid in $(pgrep -x claude 2>/dev/null); do
    grep -qz 'LOCAL_CLAUDE=1' "/proc/$pid/environ" 2>/dev/null && return 0
  done
  return 1
}
waited=0
while local_session && [ "$waited" -lt 21600 ]; do
  [ $((waited % 600)) -eq 0 ] && echo "  $(date -Is) a claude session is running; holding off"
  sleep 60; waited=$((waited + 60))
done

echo "=== $(date -Is) tuning ==="
"$HERE/local-claude" stop >/dev/null 2>&1 || true
sleep 3

{
  echo "# local-claude tuning report"
  echo "# generated $(date -Is)"
  echo "# model: $(basename "$MODEL")"
  echo "# host:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1), $(nproc) threads, $(LC_ALL=C free -g | awk '/^Mem:/{print $2}') GB RAM"
  echo
  "$HERE/test/bench.sh" --model "$MODEL" --tune
} | tee -a "$REPORT"

echo "=== $(date -Is) starting server with the tuned split ==="
# `start` brings the server up without a Claude Code session, so no throwaway
# conversation lands in the local history.
LOCAL_MODEL_FILE="$MODEL" "$HERE/local-claude" start >/dev/null 2>&1 || true

{
  echo
  echo "## API conformance ($(date -Is))"
  "$HERE/test/api_compat_test.sh"
} | tee -a "$REPORT"

echo "=== $(date -Is) done; report at $REPORT ==="
