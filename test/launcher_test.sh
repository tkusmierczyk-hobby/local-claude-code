#!/usr/bin/env bash
# launcher_test.sh — the launcher's own logic, with stand-ins for everything it drives.
#
# A fake llama-server (answers --help and --list-devices, serves /health, /props
# and /completion), a fake claude, a tiny GGUF and a scratch state folder: no
# GPU, no model, no network, and nothing outside the scratch folder is touched.
# pgrep and pkill are wrapped so the launcher only ever sees processes started
# from that folder: its `stop` falls back to pgrep, which would otherwise find
# a real server.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(mktemp -d)"
PASS=0; FAIL=0
REAL_PGREP="$(command -v pgrep)"

cleanup() {
  local pid
  for pid in $("$REAL_PGREP" -f "$ROOT"); do
    [ "$pid" = "$$" ] && continue
    grep -qF "$ROOT" "/proc/$pid/cmdline" 2>/dev/null && kill "$pid" 2>/dev/null
  done
  rm -rf "$ROOT"
}
trap cleanup EXIT

mkdir -p "$ROOT/bin" "$ROOT/llama" "$ROOT/models" "$ROOT/state"
export LOCAL_CLAUDE_STATE="$ROOT/state" MODELS_DIR="$ROOT/models" LLAMA_DIR="$ROOT/llama" \
       LOCAL_CLAUDE_CONFIG_DIR="$ROOT/config" LOCAL_SEARCH=0 ROOT REAL_PGREP
export LOCAL_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
export PATH="$ROOT/bin:$PATH"
unset LOCAL_MODEL_FILE LOCAL_NCMOE LOCAL_UBATCH LOCAL_TEMPLATE LOCAL_DEVICE LOCAL_API_KEY LLAMA_API_KEY

# ---- stand-ins -----------------------------------------------------------------
for tool in pgrep pkill; do
  cat > "$ROOT/bin/$tool" <<EOF
#!/usr/bin/env bash
# $tool, blind to every process not started from the test's scratch folder
pids=""; for pid in \$("\$REAL_PGREP" "\$@"); do
  grep -qF "\$ROOT" "/proc/\$pid/cmdline" 2>/dev/null && pids="\$pids \$pid"
done
[ -n "\$pids" ] || exit 1
EOF
done
echo 'printf "%s\n" $pids' >> "$ROOT/bin/pgrep"
echo 'kill $pids' >> "$ROOT/bin/pkill"
printf '#!/bin/sh\nexit 1\n' > "$ROOT/bin/nvidia-smi"
cat > "$ROOT/bin/claude" <<'EOF'
#!/usr/bin/env bash
# claude: records its arguments, then plays $ROOT/claude-play if there is one
printf '%s\n' "$@" > "$ROOT/claude-args"
[ -f "$ROOT/claude-play" ] && . "$ROOT/claude-play"
exit 0
EOF

printf '%s\n' "--cpu-strict <0|1>" "-lm, --load-mode MODE" "-ncmoe, --n-cpu-moe N" > "$ROOT/llama/help.txt"
cat > "$ROOT/llama/llama-server" <<'EOF'
#!/usr/bin/env bash
# llama-server: --help and --list-devices from files next to it; otherwise
# records its arguments and serves on --port until killed
D="$(dirname "$0")"
case "${1:-}" in
  --help) cat "$D/help.txt"; exit 0 ;;
  --list-devices) cat "$D/devices.txt" 2>/dev/null; exit 0 ;;
  --version) echo "version: 0 (stand-in)"; exit 0 ;;
esac
printf '%s\n' "$@" > "$D/last-args"
port=""; ctx=0; prev=""
for a in "$@"; do
  [ "$prev" = --port ] && port="$a"; [ "$prev" = --ctx-size ] && ctx="$a"; prev="$a"
done
python3 "$D/server.py" "$port" "$ctx" & child=$!
trap 'kill $child 2>/dev/null; exit 0' TERM INT
wait "$child"
EOF
cat > "$ROOT/llama/server.py" <<'EOF'
import http.server, json, sys
port, ctx = int(sys.argv[1]), int(sys.argv[2])
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == "/health":
            return self.reply({"status": "ok"})
        if self.path == "/props":
            return self.reply({"default_generation_settings": {"n_ctx": ctx}, "model_path": "stand-in"})
        self.send_error(404)
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.reply({"timings": {"prompt_n": 9500, "prompt_per_second": 600.0, "predicted_per_second": 30.0}})
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
EOF
chmod +x "$ROOT/bin/"* "$ROOT/llama/llama-server"

# A GGUF holding only metadata: a 40-layer MoE with a chat template.
python3 - "$ROOT/models/Test-A3B.gguf" <<'EOF'
import struct, sys
def s(x):
    b = x.encode(); return struct.pack("<Q", len(b)) + b
kv = [("general.architecture", 8, s("qwen35moe")), ("qwen35moe.block_count", 4, struct.pack("<I", 40)),
      ("qwen35moe.expert_count", 4, struct.pack("<I", 256)),
      ("tokenizer.chat_template", 8, s("{% for m in messages %}{{ m.content }}{% endfor %}"))]
with open(sys.argv[1], "wb") as f:
    f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)))
    for k, t, v in kv:
        f.write(s(k) + struct.pack("<I", t) + v)
EOF

check() {
  if [ "$2" = 0 ]; then echo "  ok    $1"; PASS=$((PASS+1)); else echo "  FAIL  $1"; FAIL=$((FAIL+1)); return 1; fi
}
# stop_server — stop the stand-in by the PID the launcher recorded
stop_server() {
  local f; for f in "$ROOT"/state/llama*.pid; do
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"
  done
  sleep 0.3
}

# ---- attempts ------------------------------------------------------------------
# A first attempt that stages a change, commits, and rewrites verify.sh to pass;
# a second that records the tree it started from and does the task.
P="$ROOT/proj"; mkdir -p "$P/.claude"
(
  cd "$P" && git init -q && git config user.email t@example.com && git config user.name t
  echo base > tracked.txt
  printf '#!/bin/sh\ntest -f ok\n' > .claude/verify.sh; chmod +x .claude/verify.sh
  git add -A && git commit -qm base
)
HEAD0="$(git -C "$P" rev-parse HEAD)"
cat > "$ROOT/claude-play" <<'EOF'
n=$(( $(cat "$ROOT/attempt" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$ROOT/attempt"
if [ "$n" = 1 ]; then
  echo change >> tracked.txt; git commit -qam "made by the model"
  echo staged > staged.txt; git add staged.txt
  printf '#!/bin/sh\nexit 0\n' > .claude/verify.sh
else
  { git rev-parse HEAD; git status --porcelain; } > "$ROOT/second-start"
  touch ok
fi
EOF
out="$(cd "$P" && ATTEMPT_TIMEOUT=60 "$HERE/local-claude" attempts 2 "make ok" 2>&1)"
rm -f "$ROOT/claude-play"
echo "$out" | grep -q "PASS on attempt 2"
check "attempts: an attempt cannot pass by rewriting verify.sh" $? || echo "$out" | tail -8 | sed 's/^/          /'
[ "$(cat "$ROOT/second-start" 2>/dev/null)" = "$HEAD0" ]
check "attempts: a failed attempt's commits and staged changes are undone" $?
cmp -s "$P/.claude/verify.sh" <(printf '#!/bin/sh\ntest -f ok\n')
check "attempts: verify.sh is left as it was" $?
stop_server

# ---- chat template -------------------------------------------------------------
# arg_after FLAG — the value the stand-in server got for FLAG
arg_after() { grep -A1 -x -- "$1" "$ROOT/llama/last-args" | tail -1; }
MODEL="$ROOT/models/Test-A3B.gguf"
touch -d 2000-01-01 "$MODEL"

echo "my own template" > "$ROOT/mine.jinja"; touch -d 1999-01-01 "$ROOT/mine.jinja"
LOCAL_TEMPLATE="$ROOT/mine.jinja" "$HERE/local-claude" start >/dev/null 2>&1
[ "$(cat "$ROOT/mine.jinja")" = "my own template" ] && [ "$(arg_after --chat-template-file)" = "$ROOT/mine.jinja" ]
check "template: LOCAL_TEMPLATE=<path> is used as it is, even when older than the model" $?
stop_server
LOCAL_TEMPLATE="$ROOT/missing.jinja" "$HERE/local-claude" start >/dev/null 2>&1
[ $? -ne 0 ] && [ ! -e "$ROOT/missing.jinja" ]
check "template: a LOCAL_TEMPLATE that does not exist stops the start" $?
stop_server

TPL="$ROOT/state/templates/Test-A3B.jinja"; mkdir -p "$(dirname "$TPL")"
echo stale > "$TPL"; touch -d 2001-01-01 "$TPL"   # newer than the model, older than the patch
"$HERE/local-claude" start >/dev/null 2>&1
grep -q "merge all system messages" "$TPL" && [ "$(arg_after --chat-template-file)" = "$TPL" ]
check "template: the cached copy is rebuilt when make-template.sh is newer" $?
stop_server

# ---- tuned split ---------------------------------------------------------------
# The stand-in build is called "llama" (its folder's name).
ncmoe_shown() { "$HERE/local-claude" status 2>/dev/null | sed -n 's/^n-cpu-moe: //p'; }
S="$ROOT/state"
echo 30 > "$S/ncmoe"; echo 20 > "$S/ncmoe.otherbuild"
[[ "$(ncmoe_shown)" == "40 (default"* ]]
check "tuning: another build's split is not used, shared file or not" $?
rm "$S/ncmoe.otherbuild"
[[ "$(ncmoe_shown)" == "30 (tuned)" ]]
check "tuning: the shared file of an older install still counts until a build is tuned" $?
echo 25 > "$S/ncmoe.llama"
[[ "$(ncmoe_shown)" == "25 (tuned)" ]]
check "tuning: this build's own split wins" $?
rm -f "$S"/ncmoe*

"$HERE/local-claude" bench --tune --sweep 40,38 --ubatch 2048 >/dev/null 2>&1
[ -s "$S/ncmoe.llama" ] && [ -s "$S/ubatch.llama" ] && [ ! -e "$S/ncmoe" ]
check "tuning: bench --tune stores the split for its build only" $?
stop_server

echo "launcher: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
