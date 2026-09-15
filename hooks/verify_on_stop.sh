#!/usr/bin/env bash
# Stop hook: don't let the agent declare "done" while the project's checks fail.
#
# If the project has an executable .claude/verify.sh, run it when Claude tries
# to stop. On failure, exit 2 feeds the output back to the model, which keeps
# working. Caps at 2 automatic retries per stop so a hopeless task can't loop
# forever. Only active under the local-claude launcher (LOCAL_CLAUDE=1) so it
# never changes the behavior of regular cloud Claude Code sessions.

[ "${LOCAL_CLAUDE:-}" = "1" ] || exit 0

proj="${CLAUDE_PROJECT_DIR:-$PWD}"
verify="$proj/.claude/verify.sh"
[ -x "$verify" ] || exit 0

input=$(cat)
read -r session_id active <<EOF
$(printf '%s' "$input" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('session_id','nosession'), str(d.get('stop_hook_active',False)).lower())")
EOF

counter="${TMPDIR:-/tmp}/local-claude-verify-${session_id}.count"
# A fresh stop (not one we caused by blocking) starts a new retry budget.
[ "$active" = "true" ] || echo 0 > "$counter"
n=$(cat "$counter" 2>/dev/null || echo 0)

out=$(cd "$proj" && timeout 180 bash "$verify" 2>&1)
status=$?

if [ $status -eq 0 ]; then
  rm -f "$counter"
  exit 0
fi

n=$((n + 1)); echo "$n" > "$counter"
if [ "$n" -gt 2 ]; then
  rm -f "$counter"
  printf '{"systemMessage": "verify.sh still failing after 2 automatic retries — stopping anyway. Run .claude/verify.sh manually to see failures."}\n'
  exit 0
fi

{
  echo "The project verification (.claude/verify.sh) failed (auto-retry $n/2). Output:"
  printf '%s\n' "$out" | tail -30
  echo "Fix these failures, re-run the relevant checks, and only then finish."
} >&2
exit 2
