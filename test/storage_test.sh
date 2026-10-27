#!/usr/bin/env bash
# storage_test.sh — what a real local session leaves on disk, per LOCAL_STORAGE.
#
# Needs a running server (`local-claude start`), like api_compat_test.sh. Each
# session runs with HOME, TMPDIR and the launcher's state in a scratch folder,
# so everything it writes, wherever Claude Code puts it, lands there; nothing of
# yours is read or touched. The session reads a local page with web_fetch (so
# the search server works without contacting any search engine), writes a file
# and runs a shell command. Two marker words, one in the prompt and one on the
# page, must then appear nowhere under LOCAL_STORAGE=none, and in the
# transcript by default, which shows the check can see a session's writes.
# Print mode only: an interactive session's prompt history is not covered.
#
# Usage: test/storage_test.sh    (LOCAL_PORT selects the server, default 8080)
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${LOCAL_PORT:-8080}"
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "SKIP: no server on port $PORT (start one with: local-claude start)"; exit 0
fi
MODELS="${MODELS_DIR:-$HOME/models}"
LLAMA="${LLAMA_DIR:-${LOCAL_CLAUDE_PREFIX:-$HOME/.local/share/local-claude}/current}"
ROOT="$(mktemp -d)"; SITE=""
trap '[ -n "$SITE" ] && kill "$SITE" 2>/dev/null; rm -rf "$ROOT"' EXIT
PASS=0; FAIL=0
check() {
  if [ "$2" = 0 ]; then echo "  ok    $1"; PASS=$((PASS+1)); else echo "  FAIL  $1"; FAIL=$((FAIL+1)); fi
}

ASKED="kestrel$RANDOM$RANDOM"     # only in the prompt
SHOWN="quokka$RANDOM$RANDOM"      # only on the page
mkdir -p "$ROOT/site"
echo "<html><body><p>The word is $SHOWN.</p></body></html>" > "$ROOT/site/index.html"
SITE_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
python3 -m http.server "$SITE_PORT" --bind 127.0.0.1 --directory "$ROOT/site" >/dev/null 2>&1 & SITE=$!
sleep 1

# session MODE — one session with LOCAL_STORAGE=MODE in $ROOT/MODE
session() {
  local R="$ROOT/$1"
  mkdir -p "$R/home/.local-claude" "$R/tmp" "$R/proj"
  # the config install.sh writes, its log in this run's state folder
  printf '{"mcpServers": {"search": {"command": "%s", "args": ["%s"], "env": {"SEARCH_MCP_LOG": "%s"}}}}' \
    "$(command -v python3)" "$HERE/mcp/search_server.py" "$R/home/.local-claude/search-mcp.log" \
    > "$R/home/.local-claude/search-mcp.json"
  (cd "$R/proj" && env -u XDG_CACHE_HOME -u XDG_CONFIG_HOME -u XDG_DATA_HOME -u XDG_STATE_HOME \
      -u LOCAL_CLAUDE_STATE -u LOCAL_CLAUDE_CONFIG_DIR -u CLAUDE_CONFIG_DIR \
      HOME="$R/home" TMPDIR="$R/tmp" MODELS_DIR="$MODELS" LLAMA_DIR="$LLAMA" LOCAL_PORT="$PORT" \
      LOCAL_STORAGE="$1" LOCAL_SEARCH=1 WEB_FETCH_ALLOW_PRIVATE=1 \
      timeout 900 "$HERE/local-claude" --dangerously-skip-permissions -p \
      "Use the web_fetch tool to read http://127.0.0.1:$SITE_PORT/ and find the word on that page. Create a file note.txt containing that word. Then run the shell command: cat note.txt. Finally reply with the single word DONE. (Reference: $ASKED)" \
      </dev/null >"$R/reply" 2>"$R/stderr")
  [ -s "$R/proj/note.txt" ] && grep -q "$SHOWN" "$R/proj/note.txt"
}
# left MODE — the files the session left outside its project folder
left() { (cd "$ROOT/$1" && find home tmp -type f ! -name search-mcp.json | sort); }
# mentions MODE — files outside the project that hold either marker word
mentions() { (cd "$ROOT/$1" && grep -rlE "$ASKED|$SHOWN" home tmp --exclude=search-mcp.json 2>/dev/null | sort); }

echo "server on port $PORT; one short session per mode (a minute or two each)"
session separate
check "default: the session fetched the page, wrote the file and ran the command" $?
left separate | grep -qE '^home/\.local-claude/claude/projects/.*\.jsonl$' && [ -n "$(mentions separate)" ]
check "default: the transcript holds the conversation (so the check below can see writes)" $?
[ -s "$ROOT/separate/home/.local-claude/search-mcp.log" ]
check "default: the search server logs which tool ran" $?

session none
check "none: the session fetched the page, wrote the file and ran the command" $?
found="$(mentions none)"
[ -z "$found" ]
check "none: no file outside the project holds anything from the conversation" $? || echo "$found" | sed 's/^/          /'
! left none | grep -qE '^home/\.local-claude/claude/(projects/.*\.jsonl|history\.jsonl|file-history/|projects/.*/memory/)'
check "none: no transcript, prompt history, file snapshots or memory" $?
[ ! -e "$ROOT/none/home/.local-claude/search-mcp.log" ]
check "none: no search log" $?
echo "  --    still written under none (Claude Code's own housekeeping and connection log):"
left none | sed -E 's#claude-cli-nodejs/[^/]+#claude-cli-nodejs/<project>#; s#[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}#<id>#g; s#^#          #'

echo "storage: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
