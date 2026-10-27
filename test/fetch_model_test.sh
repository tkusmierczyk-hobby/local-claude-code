#!/usr/bin/env bash
# fetch_model_test.sh — bin/fetch-model.sh against a local stand-in for Hugging Face.
#
# The downloader runs unattended for many hours on a slow link, so the failures
# it has to survive are faked here rather than met for the first time halfway
# through a 22 GB model: error pages, dropped connections, a size it cannot learn.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"; SERVER_PID=""
trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; rm -rf "$TMP"' EXIT
PASS=0; FAIL=0

# The stand-in serves the tree API and resolve/main/model.gguf with byte ranges.
# Files in its folder steer it: api_fail and dl_fail hold how many requests to
# answer with HTTP 503 (with an HTML page, as real servers do), chunk how many
# bytes to send before dropping the connection.
mkdir -p "$TMP/srv"
cat > "$TMP/server.py" <<'PY'
import http.server, json, os, re, sys
root = sys.argv[1]

def ctl(name):
    try:
        with open(os.path.join(root, name)) as fh:
            return int(fh.read().strip() or 0)
    except OSError:
        return 0

def take(name):
    n = ctl(name)
    if n > 0:
        with open(os.path.join(root, name), "w") as fh:
            fh.write(str(n - 1))
    return n > 0

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        with open(os.path.join(root, "payload"), "rb") as fh:
            data = fh.read()
        if self.path.startswith("/api/models/"):
            if take("api_fail"):
                return self.send_error(503)
            body = json.dumps([{"type": "file", "path": "model.gguf", "size": len(data)}]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        if not self.path.endswith("/resolve/main/model.gguf"):
            return self.send_error(404)
        if take("dl_fail"):
            return self.send_error(503)
        m = re.match(r"bytes=(\d+)-", self.headers.get("Range", ""))
        start = int(m.group(1)) if m else 0
        if start >= len(data):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(data)}")
            self.send_header("Content-Length", "0")
            return self.end_headers()
        self.send_response(206 if m else 200)
        if m:
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:start + (ctl("chunk") or len(data))])

srv = http.server.HTTPServer(("127.0.0.1", 0), H)
with open(os.path.join(root, "port"), "w") as fh:
    fh.write(str(srv.server_address[1]))
srv.serve_forever()
PY
head -c 100000 /dev/urandom > "$TMP/srv/payload"
SHA="$(sha256sum "$TMP/srv/payload" | cut -d' ' -f1)"
python3 "$TMP/server.py" "$TMP/srv" & SERVER_PID=$!
for _ in $(seq 1 50); do [ -s "$TMP/srv/port" ] && break; sleep 0.1; done
PORT="$(cat "$TMP/srv/port")"

# fetch <file> [sha256] — one download into a fresh models folder; sets RC.
# The timeout catches a downloader that never finishes.
fetch() {
  rm -rf "$TMP/models" "$TMP/fetch.log"
  MODELS_DIR="$TMP/models" FETCH_LOG="$TMP/fetch.log" FETCH_PAUSE=0 \
    HF_ENDPOINT="http://127.0.0.1:$PORT" timeout 60 "$HERE/bin/fetch-model.sh" test/repo "$@"
  RC=$?
}
# queue <name>=<n>... — set the stand-in's failure counters, all others to 0
queue() {
  local kv; for kv in api_fail=0 dl_fail=0 chunk=0 "$@"; do echo "${kv#*=}" > "$TMP/srv/${kv%%=*}"; done
}
check() {
  if [ "$2" = 0 ]; then echo "  ok    $1"; PASS=$((PASS+1))
  else echo "  FAIL  $1"; sed 's/^/          /' "$TMP/fetch.log" | tail -5; FAIL=$((FAIL+1)); fi
}
same() { [ "$RC" = 0 ] && cmp -s "$TMP/models/model.gguf" "$TMP/srv/payload"; }

queue
fetch model.gguf "$SHA"; same
check "a plain download arrives intact" $?

queue dl_fail=1
fetch model.gguf "$SHA"; same
check "an error page answering the first request is not saved into the file" $?

queue chunk=30000
fetch model.gguf "$SHA"; same
check "dropped connections resume where the file got to" $?

echo "fetch-model: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
