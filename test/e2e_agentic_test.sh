#!/usr/bin/env bash
# End-to-end test: Claude Code CLI → local llama-server → GPU + CPU-offloaded MoE,
# exercising the full agentic loop (Write tool + Bash tool) in a sandbox.
# Usage: ./e2e_agentic_test.sh [model-file]   (default: whatever local-claude picks)
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-}"
SANDBOX=$(mktemp -d /tmp/local-claude-e2e.XXXXXX)
# The test session gets its own Claude Code folder, deleted afterwards, so it
# never appears in your local history. An explicit LOCAL_CLAUDE_CONFIG_DIR is kept.
if [ -z "${LOCAL_CLAUDE_CONFIG_DIR:-}" ]; then
  LOCAL_CLAUDE_CONFIG_DIR=$(mktemp -d /tmp/local-claude-e2e-config.XXXXXX)
  trap 'rm -rf "$LOCAL_CLAUDE_CONFIG_DIR"' EXIT
fi
export LOCAL_CLAUDE_CONFIG_DIR
cd "$SANDBOX"
echo "sandbox: $SANDBOX  model: ${MODEL:-auto}"

START=$(date +%s)
LOCAL_MODEL_FILE="$MODEL" "$HERE/local-claude" \
  --dangerously-skip-permissions \
  -p "Create a file fib.py containing a function fib(n) returning the n-th Fibonacci number (fib(0)=0, fib(1)=1), plus a __main__ block that prints fib(10). Run it with python3 to verify it prints 55. If the output is wrong, fix fib.py and re-run until it prints 55. Then reply exactly DONE." \
  </dev/null 2>&1 | tail -20
ELAPSED=$(( $(date +%s) - START ))
echo "--- elapsed: ${ELAPSED}s ---"

echo "== checks =="
FAIL=0
[ -f fib.py ] && echo "PASS: fib.py exists" || { echo "FAIL: fib.py missing"; FAIL=1; }
if [ -f fib.py ]; then
  OUT=$(timeout 10 python3 fib.py 2>&1)
  [ "$OUT" = "55" ] && echo "PASS: fib.py prints 55" || { echo "FAIL: fib.py printed: $OUT"; FAIL=1; }
fi
echo "sandbox kept at $SANDBOX"
exit $FAIL
