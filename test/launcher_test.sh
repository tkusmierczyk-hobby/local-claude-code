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
# $tool, blind to every process not started from the test's scratch folder,
# itself included (its own command line holds the pattern)
pids=""; for pid in \$("\$REAL_PGREP" "\$@"); do
  [ "\$pid" = "\$\$" ] && continue
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
port=""; ctx=0; ub=0; vision=0; prev=""
for a in "$@"; do
  [ "$prev" = --port ] && port="$a"; [ "$prev" = --ctx-size ] && ctx="$a"
  [ "$prev" = --ubatch-size ] && ub="$a"; [ "$a" = --mmproj ] && vision=1; prev="$a"
done
# FAKE_MAX_UBATCH=N: fail as a start that runs out of VRAM does, above that micro-batch
if [ -n "${FAKE_MAX_UBATCH:-}" ] && [ "$ub" -gt "$FAKE_MAX_UBATCH" ]; then
  echo "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1200 MiB on device 0: cudaMalloc failed: out of memory" >&2
  echo "llama_init_from_model: failed to allocate compute buffers" >&2
  exit 1
fi
trap 'kill $child 2>/dev/null; exit 0' TERM INT
# FAKE_IGNORE_TERM=1: shrug off SIGTERM, so that only SIGKILL stops it
[ -n "${FAKE_IGNORE_TERM:-}" ] && trap '' TERM
# FAKE_SLOW_START=N: take N seconds to load, as a large model does from disk
if [ -n "${FAKE_SLOW_START:-}" ]; then sleep "$FAKE_SLOW_START" & child=$!; wait "$child"; fi
# SIGUSR1: end as llama.cpp does on a CUDA error, with its message and SIGABRT's status
trap 'echo "CUDA error: out of memory" >&2; kill $child 2>/dev/null; exit 134' USR1
python3 "$D/server.py" "$port" "$ctx" "$vision" & child=$!
wait "$child"
EOF
cat > "$ROOT/llama/server.py" <<'EOF'
import http.server, json, os, sys, threading, time
port, ctx, vision = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
# Go with the llama-server script, even when it is killed with SIGKILL.
def orphaned(parent=os.getppid()):
    while os.getppid() == parent:
        time.sleep(0.1)
    os._exit(0)
threading.Thread(target=orphaned, daemon=True).start()
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
            return self.reply({"default_generation_settings": {"n_ctx": ctx}, "model_path": "stand-in",
                               "modalities": {"vision": vision}})
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

# ---- vision --------------------------------------------------------------------
# Without a projector the server rejects every image, and Claude Code shows only
# an API error, so the start and status say so.
out="$("$HERE/local-claude" start 2>&1 >/dev/null)"
grep -q "vision off (no mmproj\*.gguf in $ROOT/models): requests with an image will fail" <<<"$out" \
  && ! grep -qx -- --mmproj "$ROOT/llama/last-args" \
  && grep -q "^vision:   off (no mmproj" <<<"$("$HERE/local-claude" status 2>/dev/null)"
check "vision: without a projector, start and status say images will fail" $? || echo "$out" | sed 's/^/          /'
touch "$ROOT/models/mmproj-test.gguf"
grep -q "^vision:   off — mmproj-test.gguf is there now; restart the server" <<<"$("$HERE/local-claude" status 2>/dev/null)"
check "vision: status says a projector downloaded since the start needs a restart" $?
stop_server
out="$("$HERE/local-claude" start 2>&1 >/dev/null)"
[ "$(arg_after --mmproj)" = "$ROOT/models/mmproj-test.gguf" ] && grep -qx -- --no-mmproj-offload "$ROOT/llama/last-args" \
  && grep -q "+vision(mmproj-test.gguf)" <<<"$out" && grep -qx "vision:   on" <<<"$("$HERE/local-claude" status 2>/dev/null)"
check "vision: the one projector in the models folder is loaded, on the CPU" $?
stop_server
touch "$ROOT/models/mmproj-other.gguf"
out="$("$HERE/local-claude" start 2>&1 >/dev/null)"
grep -q "vision off (several mmproj files" <<<"$out" && ! grep -qx -- --mmproj "$ROOT/llama/last-args"
check "vision: with several projectors, none is guessed" $?
stop_server
LOCAL_MMPROJ=mmproj-other.gguf "$HERE/local-claude" start >/dev/null 2>&1
[ "$(arg_after --mmproj)" = "$ROOT/models/mmproj-other.gguf" ]
check "vision: LOCAL_MMPROJ picks one by its name in the models folder" $?
stop_server
out="$(LOCAL_MMPROJ=0 "$HERE/local-claude" start 2>&1 >/dev/null)"
! grep -q "vision" <<<"$out" && ! grep -qx -- --mmproj "$ROOT/llama/last-args" \
  && grep -qx "vision:   off" <<<"$(LOCAL_MMPROJ=0 "$HERE/local-claude" status 2>/dev/null)"
check "vision: LOCAL_MMPROJ=0 turns it off without a warning" $?
stop_server
rm -f "$ROOT/models/mmproj-"*.gguf

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

echo 37 > "$S/ncmoe.llama"; echo 4096 > "$S/ubatch.llama"
row="$("$HERE/local-claude" bench 2>/dev/null | grep -E '^[0-9]' | head -1)"
[ "$(arg_after --n-cpu-moe) $(arg_after --ubatch-size)" = "37 4096" ] && [[ "$row" == "37 "*" 4096 "* ]]
check "bench: without --tune it measures the split the launcher runs" $? || echo "          got: $row"
stop_server; rm -f "$S"/ncmoe* "$S"/ubatch*

# ---- search backend ------------------------------------------------------------
# The config install.sh writes, with the requests-based server.
cat > "$S/search-mcp.json" <<EOF
{"mcpServers": {"search": {"command": "$(command -v python3)", "args": ["$HERE/mcp/search_server.py"],
 "env": {"SEARCH_MCP_LOG": "$S/search-mcp.log"}}}}
EOF
printf '#!/bin/sh\nexit 0\n' > "$ROOT/bin/lynx"; chmod +x "$ROOT/bin/lynx"
# session_search [env...] — the search server script a session gets
session_search() {
  rm -f "$ROOT/claude-args"
  env "$@" LOCAL_SEARCH=1 "$HERE/local-claude" -p hi </dev/null >/dev/null 2>&1
  local cfg; cfg="$(grep -A1 -x -- --mcp-config "$ROOT/claude-args" 2>/dev/null | tail -1)"
  [ -n "$cfg" ] && python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["mcpServers"]["search"]["args"][0])' "$cfg"
}
[[ "$(session_search)" == */search_server.py ]]
check "search: without LOCAL_SEARCH_BACKEND, the installed server" $?
[[ "$(session_search LOCAL_SEARCH_BACKEND=lynx)" == */search_server_lynx.py ]]
check "search: LOCAL_SEARCH_BACKEND=lynx switches this session to the lynx server" $?
[[ "$(LOCAL_SEARCH=1 LOCAL_SEARCH_BACKEND=lynx "$HERE/local-claude" status 2>/dev/null | grep '^search:')" == "search:   lynx "* ]] \
  && [[ "$(LOCAL_SEARCH=1 "$HERE/local-claude" status 2>/dev/null | grep '^search:')" == "search:   requests "* ]]
sed "s|search_server.py|search_server_lynx.py|" "$S/search-mcp.json" > "$ROOT/lynx-mcp.json"
[[ "$(LOCAL_SEARCH=1 LOCAL_SEARCH_CONFIG="$ROOT/lynx-mcp.json" "$HERE/local-claude" status 2>/dev/null | grep '^search:')" == "search:   lynx "* ]]
check "search: status names the backend sessions will get" $?
stop_server
LOCAL_SEARCH=1 LOCAL_SEARCH_BACKEND=links "$HERE/local-claude" -p hi </dev/null >/dev/null 2>&1
[ $? -ne 0 ] && [ ! -e "$S/llama.pid" ]
check "search: an unknown LOCAL_SEARCH_BACKEND fails before the model loads" $?
stop_server

# ---- GPU choice ----------------------------------------------------------------
# device_for <--list-devices lines>... — the device the launcher picks
device_for() {
  { echo "Available devices:"; printf '  %s\n' "$@"; } > "$ROOT/llama/devices.txt"
  rm -f "$S"/device.*
  "$HERE/local-claude" status 2>/dev/null | sed -n 's/^device: *//p'
}
[ "$(device_for "Vulkan0: Intel(R) Graphics (RPL-S) (23443 MiB, 20052 MiB free)" \
                "Vulkan1: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB, 460 MiB free)")" = Vulkan1 ]
check "gpu: the NVIDIA card, not the integrated GPU listed first (reference machine)" $?
[ "$(device_for "Vulkan0: Intel(R) Arc(TM) Graphics (MTL) (18000 MiB, 17000 MiB free)" \
                "Vulkan1: NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB, 8000 MiB free)")" = Vulkan1 ]
check "gpu: an integrated Intel Arc listed first is passed over" $?
[ "$(device_for "Vulkan0: Intel(R) UHD Graphics 770 (16000 MiB, 15000 MiB free)" \
                "Vulkan1: Intel(R) Arc(TM) A770 Graphics (16384 MiB, 16000 MiB free)")" = Vulkan1 ]
check "gpu: a discrete Intel Arc is chosen" $?
[ -z "$(device_for "Vulkan0: Intel(R) Arc(TM) 140V GPU (16GB) (18000 MiB, 17000 MiB free)")" ]
check "gpu: an integrated GPU alone is left to llama.cpp" $?
rm -f "$ROOT/llama/devices.txt" "$S"/device.*

# ---- stop ----------------------------------------------------------------------
printf '#!/usr/bin/env bash\nwhile :; do sleep 1; done\n' > "$ROOT/bin/ollama"; chmod +x "$ROOT/bin/ollama"
"$ROOT/bin/ollama" serve & ours=$!
"$ROOT/bin/ollama" serve & yours=$!
echo "$ours" > "$S/ollama.pid"
"$HERE/local-claude" stop >/dev/null 2>&1; sleep 0.3
! kill -0 "$ours" 2>/dev/null && kill -0 "$yours" 2>/dev/null
check "stop: only the ollama the launcher started is stopped, not one of yours" $?
kill "$ours" "$yours" 2>/dev/null; wait "$ours" "$yours" 2>/dev/null

# Two servers, as with a test server beside the main one.
free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
A="$LOCAL_PORT"; B="$(free_port)"
up() { curl -sf "http://127.0.0.1:$1/health" >/dev/null; }
LOCAL_PORT=$A "$HERE/local-claude" start >/dev/null 2>&1
LOCAL_PORT=$B "$HERE/local-claude" start >/dev/null 2>&1
LOCAL_PORT=$A "$HERE/local-claude" stop >/dev/null 2>&1
! up "$A" && up "$B"
check "stop: stops the server on its own port, not the one started last" $?
LOCAL_PORT=$A "$HERE/local-claude" start >/dev/null 2>&1
rm -f "$S"/llama*.pid   # e.g. state from before an upgrade
LOCAL_PORT=$B "$HERE/local-claude" stop >/dev/null 2>&1
up "$A" && ! up "$B"
check "stop: without a PID file, only the server on that port" $?
LOCAL_PORT=$A "$HERE/local-claude" stop >/dev/null 2>&1
bash -c 'while :; do sleep 1; done' & other=$!
for f in llama.pid "llama.$A.pid" "llama.$B.pid"; do echo "$other" > "$S/$f"; done
LOCAL_PORT=$B "$HERE/local-claude" stop >/dev/null 2>&1; sleep 0.3
kill -0 "$other" 2>/dev/null
check "stop: a recorded PID that is no longer a llama-server is left alone" $?
kill "$other" 2>/dev/null; wait "$other" 2>/dev/null; stop_server

# ---- locale --------------------------------------------------------------------
# `free` labels its rows in the user's language ("Pamięć:" in Polish).
"$HERE/local-claude" status >/dev/null 2>&1   # the stand-in nvidia-smi fails, as on a machine without one
check "status: completes on a machine without nvidia-smi" $?
LC_ALL=pl_PL.UTF-8 "$HERE/local-claude" status 2>/dev/null | grep -q '^ram: '
check "locale: status reports RAM whatever the language" $?

# ---- init ----------------------------------------------------------------------
for mk in '.PHONY: all' 'build:'; do
  I="$ROOT/init"; rm -rf "$I"; mkdir -p "$I"; printf '%s\n' "$mk" > "$I/Makefile"
  (cd "$I" && "$HERE/local-claude" init >/dev/null 2>&1) && [ -s "$I/CLAUDE.md" ] && [ -f "$I/.claude/verify.sh.example" ]
  check "init: scaffolds everything for a Makefile holding '$mk'" $?
done
grep -q "Makefile targets: build" "$I/CLAUDE.md"
check "init: lists the Makefile's targets" $?

# ---- syntax-check hook (installed by init) -------------------------------------
H="$ROOT/hook"; mkdir -p "$H"
printf 'def f(:\n' > "$H/bad.py"; printf 'x = 1\n' > "$H/good.py"
printf '{\n  // TypeScript allows comments\n  "compilerOptions": {"baseUrl": "http://x//y", "strict": true,},\n}\n' > "$H/tsconfig.json"
printf '{"a": }\n' > "$H/bad.json"
# hook FILE [env...] — the hook's exit status for an edit of FILE
hook() { local f="$1"; shift; printf '{"tool_input": {"file_path": "%s"}}' "$f" | env "$@" "$HERE/hooks/syntax_check.sh" >/dev/null 2>&1; echo $?; }
[ "$(hook "$H/bad.py" -u LOCAL_CLAUDE)" = 0 ]
check "hook: a cloud session (no LOCAL_CLAUDE=1) is left alone" $?
[ "$(hook "$H/bad.py" LOCAL_CLAUDE=1)" = 2 ] && [ "$(hook "$H/bad.json" LOCAL_CLAUDE=1)" = 2 ]
check "hook: broken Python and JSON are sent back" $?
[ "$(hook "$H/good.py" LOCAL_CLAUDE=1)" = 0 ] && [ ! -e "$H/__pycache__" ]
check "hook: checking Python leaves no __pycache__ in the project" $?
[ "$(hook "$H/tsconfig.json" LOCAL_CLAUDE=1)" = 0 ]
check "hook: JSON with comments and trailing commas (tsconfig.json) passes" $?

# ---- auto-tune -----------------------------------------------------------------
rm -f "$ROOT/claude-args" "$S"/ncmoe* "$S"/ubatch*
timeout 300 "$HERE/bin/auto-tune.sh" "$ROOT/models/Test-A3B.gguf" </dev/null >/dev/null 2>&1
grep -q "^tuned:" "$S/bench-results.txt" 2>/dev/null && [ -s "$S/auto-tune.log" ] && [ -s "$S/ncmoe.llama" ]
check "auto-tune: tunes, and keeps its report and log in LOCAL_CLAUDE_STATE" $?
curl -sf "http://127.0.0.1:$LOCAL_PORT/health" >/dev/null && [ ! -e "$ROOT/claude-args" ]
check "auto-tune: leaves the server running without starting a Claude Code session" $?
stop_server; rm -f "$S"/ncmoe* "$S"/ubatch*

# ---- start timeout -------------------------------------------------------------
out="$(LOCAL_START_TIMEOUT=2 FAKE_SLOW_START=30 "$HERE/local-claude" start 2>&1)"; rc=$?
sleep 0.5
[ "$rc" != 0 ] && echo "$out" | grep -q "not ready after 2s" && ! pgrep -f "llama-server --model" >/dev/null
check "start: a server not ready in time is stopped, not left loading" $? || echo "$out" | tail -3 | sed 's/^/          /'
"$HERE/local-claude" start >/dev/null 2>&1 && curl -sf "http://127.0.0.1:$LOCAL_PORT/health" >/dev/null
check "start: the next start then works" $?
stop_server

# ---- the server log under LOCAL_STORAGE=none -------------------------------------
# (The search log is the search server's own rule: test/mcp_parity_test.py.)
logs() { ls "$S"/llama-server*.log 2>/dev/null | wc -l; }
rm -f "$S"/llama-server*.log
"$HERE/local-claude" start >/dev/null 2>&1; [ "$(logs)" = 1 ]
check "storage: the server keeps a log by default" $?
stop_server; rm -f "$S"/llama-server*.log
LOCAL_STORAGE=none "$HERE/local-claude" start >/dev/null 2>&1 && up "$LOCAL_PORT" && [ "$(logs)" = 0 ]
check "storage: a server started under LOCAL_STORAGE=none keeps none" $?
stop_server
LOCAL_STORAGE=none LOCAL_SERVER_LOG=1 "$HERE/local-claude" start >/dev/null 2>&1; [ "$(logs)" = 1 ]
check "storage: LOCAL_SERVER_LOG=1 keeps it anyway" $?
stop_server; rm -f "$S"/llama-server*.log
echo 4096 > "$S/ubatch.llama"   # a tuned micro-batch that no longer fits
for mode in separate none; do
  out="$(FAKE_MAX_UBATCH=2048 LOCAL_STORAGE=$mode "$HERE/local-claude" start 2>&1)"
  echo "$out" | grep -q "retrying at 2048" && up "$LOCAL_PORT" && [ "$(arg_after --ubatch-size)" = 2048 ]
  check "storage: the out-of-VRAM retry at 2048 works with LOCAL_STORAGE=$mode" $? || echo "$out" | tail -3 | sed 's/^/          /'
  [ "$mode" = none ] && { [ "$(logs)" = 0 ]; check "storage: ...and leaves no log behind" $?; }
  stop_server; rm -f "$S"/llama-server*.log
done
rm -f "$S/ubatch.llama"

# ---- restart after a crash -------------------------------------------------------
PIDF="$S/llama.$LOCAL_PORT.pid" WATCHF="$S/llama.$LOCAL_PORT.watch" SLOG="$S/llama-server.$LOCAL_PORT.log"
watcher_pid() { cut -d' ' -f1 "$WATCHF"; }
# restarted OLD_PID — a server other than OLD_PID comes up within 15 s
restarted() {
  local new
  for _ in $(seq 1 150); do
    new="$(cat "$PIDF" 2>/dev/null)"
    [ -n "$new" ] && [ "$new" != "$1" ] && up "$LOCAL_PORT" && return 0
    sleep 0.1
  done
  return 1
}
# settled WATCHER — that watcher has finished restarting the server, within 15 s
settled() { for _ in $(seq 1 150); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.1; done; return 1; }
# stays_down — no server comes back (a restart would be up within ~4 s)
stays_down() { sleep 5; ! up "$LOCAL_PORT"; }
"$HERE/local-claude" stop >/dev/null 2>&1; rm -f "$S"/llama-server*.log

timeout 20 bash -c 'out="$("$1" start 2>&1)"' _ "$HERE/local-claude"
check "restart: the watcher leaves a caller's \$(...) free to return" $?
old="$(cat "$PIDF")"; kill -KILL "$old"
restarted "$old" && grep -q "SIGKILL.*Restarting" "$SLOG"
check "restart: a server killed with SIGKILL, as when RAM runs out, is started again" $?
old="$(cat "$PIDF")"; kill "$old"
stays_down
check "restart: a server stopped with a plain kill stays stopped" $?

echo 4096 > "$S/ubatch.llama"
"$HERE/local-claude" start >/dev/null 2>&1; old="$(cat "$PIDF")"; kill -USR1 "$old"
restarted "$old" && [ "$(arg_after --ubatch-size)" = 2048 ] && grep -q "GPU ran out of memory" "$SLOG"
check "restart: after the GPU ran out of memory, at --ubatch-size 2048 instead of the tuned 4096" $?
watcher="$(watcher_pid)"; "$HERE/local-claude" stop >/dev/null 2>&1
stays_down && ! kill -0 "$watcher" 2>/dev/null
check "restart: local-claude stop ends the watcher too" $?
ln -sf "$HERE/local-claude" "$ROOT/bin/lc"
FAKE_IGNORE_TERM=1 "$ROOT/bin/lc" start >/dev/null 2>&1; watcher="$(watcher_pid)"
LOCAL_STOP_TIMEOUT=2 "$ROOT/bin/lc" stop >/dev/null 2>&1
stays_down && ! kill -0 "$watcher" 2>/dev/null
check "restart: stop ends the watcher under any launcher name, even when the server needs SIGKILL" $?
"$HERE/local-claude" stop >/dev/null 2>&1
rm -f "$S"/llama-server*.log
LOCAL_STORAGE=none "$HERE/local-claude" start >/dev/null 2>&1; old="$(cat "$PIDF")"; kill -USR1 "$old"
restarted "$old" && [ "$(arg_after --ubatch-size)" = 2048 ] && sleep 1 && [ "$(logs)" = 0 ] \
  && grep -q "GPU ran out of memory" "$S/llama.$LOCAL_PORT.crash"
check "restart: under LOCAL_STORAGE=none it still reads why, and leaves no log behind" $?
"$HERE/local-claude" stop >/dev/null 2>&1; rm -f "$S/ubatch.llama"

LOCAL_RESTART=1 "$HERE/local-claude" start >/dev/null 2>&1; old="$(cat "$PIDF")" watcher="$(watcher_pid)"
kill -KILL "$old"
restarted "$old" && settled "$watcher" && kill -KILL "$(cat "$PIDF")" && stays_down && grep -q "Not restarted: crash 2 in 10 minutes" "$SLOG"
check "restart: LOCAL_RESTART caps the restarts, so a server that keeps crashing stays down" $? \
  || grep -a '^\[local-claude\]' "$SLOG" | tail -3 | sed 's/^/          /'
LOCAL_RESTART=0 "$HERE/local-claude" start >/dev/null 2>&1; kill -KILL "$(cat "$PIDF")"
stays_down && grep -q "Not restarted (LOCAL_RESTART=0)" "$SLOG"
check "restart: LOCAL_RESTART=0 never restarts" $?
grep -q "^last crash: .*LOCAL_RESTART=0" <<< "$("$HERE/local-claude" status 2>/dev/null)"
check "restart: status shows the latest crash" $?

# statusline_cmd [env...] — the status line command a session gets; the session
# starts the server if it is down
statusline_cmd() {
  rm -f "$ROOT/claude-args"
  env "$@" "$HERE/local-claude" -p hi </dev/null >/dev/null 2>&1
  grep -A1 -x -- --settings "$ROOT/claude-args" | tail -1 \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("statusLine",{}).get("command",""))'
}
# settings [args...] — the --settings a session gets, as one JSON line
settings() {
  rm -f "$ROOT/claude-args"
  "$HERE/local-claude" -p hi "$@" </dev/null >/dev/null 2>&1
  grep -A1 -x -- --settings "$ROOT/claude-args" | tail -1
}
# has JSON PYTHON-EXPRESSION — the expression holds for the settings s
has() { python3 -c 'import json,sys; s=json.loads(sys.argv[1]); sys.exit(0 if eval(sys.argv[2]) else 1)' "$1" "$2"; }
has "$(settings --settings '{"theme": "light", "skipWebFetchPreflight": false}')" \
  's["theme"] == "light" and s["skipWebFetchPreflight"] is False and s["autoUploadSessions"] is False and len(s["claudeMdExcludes"]) == 1 and "statusLine" in s'
check "settings: your --settings are merged with the launcher's, yours winning key by key" $?
echo '{"claudeMdExcludes": ["/elsewhere/CLAUDE.md"], "statusLine": {"type": "command", "command": "echo mine"}}' > "$ROOT/mine.json"
has "$(settings --settings "$ROOT/mine.json" --settings='{"theme": "dark"}')" \
  'len(s["claudeMdExcludes"]) == 2 and s["statusLine"]["command"] == "echo mine" and s["theme"] == "dark"'
check "settings: --settings files and --settings= work too, several at once" $?
# age FILE MINUTES — set the file's time that many minutes back
age() { python3 -c 'import os,sys,time; t=time.time()-60*float(sys.argv[2]); os.utime(sys.argv[1],(t,t))' "$1" "$2"; }
NOTE="$S/llama.$LOCAL_PORT.crash"
SL="$(statusline_cmd)"
[ -n "$SL" ] && [ -z "$(sh -c "$SL")" ]
check "status line: a crash before the latest start is not shown" $?
watcher="$(watcher_pid)" old="$(cat "$PIDF")"; kill -KILL "$old"
restarted "$old" && settled "$watcher" \
  && [[ "$(sh -c "$SL")" == [0-9][0-9]:[0-9][0-9]:[0-9][0-9]" llama-server stopped: it was killed (SIGKILL)"*". Restarted." ]]
check "status line: after a restart, sessions show when and why the server crashed, and that it is back" $? \
  || sed 's/^/          /' "$NOTE"
age "$PIDF" 20; age "$NOTE" 5; shown="$(sh -c "$SL")"; age "$NOTE" 11
[ -n "$shown" ] && [ -z "$(sh -c "$SL")" ]
check "status line: the crash is shown for 10 minutes" $?
[ -z "$(statusline_cmd LOCAL_NOTIFY=0)" ]
check "status line: LOCAL_NOTIFY=0 leaves it out" $?
[ "$(statusline_cmd LOCAL_STORAGE=none)" = "$SL" ]
check "status line: LOCAL_STORAGE=none keeps it, as it shows no part of the conversation" $?
mkdir -p "$ROOT/config"; [ -f "$ROOT/config/settings.json" ] && cp "$ROOT/config/settings.json" "$ROOT/settings.bak"
echo '{"statusLine": {"type": "command", "command": "echo mine"}}' > "$ROOT/config/settings.json"
[ -z "$(statusline_cmd)" ]
check "status line: a status line of your own wins" $?
rm -f "$ROOT/config/settings.json"; [ -f "$ROOT/settings.bak" ] && mv "$ROOT/settings.bak" "$ROOT/config/settings.json"
"$HERE/local-claude" stop >/dev/null 2>&1

echo "launcher: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
