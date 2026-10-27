#!/usr/bin/env bash
# install_test.sh — install.sh against a stand-in for GitHub's API and downloads.
#
# Installs into a scratch folder (--no-symlink, scratch prefix, state and
# models), from two fake release archives whose llama-server only reports
# which backend it is. curl and claude are stand-ins; nothing is downloaded.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
[ "$(uname -s)/$(uname -m)" = Linux/x86_64 ] || { echo "SKIP: the fake releases are Linux x64 builds"; exit 0; }
ROOT="$(mktemp -d)"; trap 'rm -rf "$ROOT"' EXIT
PASS=0; FAIL=0
mkdir -p "$ROOT/bin" "$ROOT/rel"

# Two release archives, laid out like llama.cpp's (build/bin/...).
for kind in vulkan cpu; do
  mkdir -p "$ROOT/pkg-$kind/build/bin"
  printf '#!/bin/sh\n[ "$1" = --version ] && echo "version: 9000 (%s)"\n[ "$1" = --help ] && echo "--n-cpu-moe N"\nexit 0\n' "$kind" \
    > "$ROOT/pkg-$kind/build/bin/llama-server"
  chmod +x "$ROOT/pkg-$kind/build/bin/llama-server"
  tar czf "$ROOT/rel/$kind.tar.gz" -C "$ROOT/pkg-$kind" build
done
python3 - "$ROOT/rel" > "$ROOT/rel/api.json" <<'EOF'
import hashlib, json, os, sys
d = sys.argv[1]
def asset(kind, name):
    data = open(os.path.join(d, kind + ".tar.gz"), "rb").read()
    return {"name": name, "browser_download_url": "https://example.invalid/" + kind + ".tar.gz",
            "size": len(data), "digest": "sha256:" + hashlib.sha256(data).hexdigest()}
print(json.dumps([{"tag_name": "b9000", "assets": [
    asset("vulkan", "llama-b9000-bin-ubuntu-vulkan-x64.tar.gz"),
    asset("cpu", "llama-b9000-bin-ubuntu-x64.tar.gz")]}]))
EOF
cat > "$ROOT/bin/curl" <<'EOF'
#!/usr/bin/env bash
# curl stand-in: the release list for api.github.com, the archives for example.invalid
out=""; prev=""; for a in "$@"; do [ "$prev" = -o ] && out="$a"; prev="$a"; done
url="${@: -1}"
case "$url" in
  https://api.github.com/*) cat "$REL/api.json" ;;
  https://example.invalid/*) cp "$REL/${url##*/}" "$out" ;;
  *) exit 6 ;;
esac
EOF
printf '#!/bin/sh\necho "2.0.0 (stand-in)"\n' > "$ROOT/bin/claude"
chmod +x "$ROOT/bin/"*
export PATH="$ROOT/bin:$PATH" REL="$ROOT/rel"
unset LOCAL_SEARCH_BACKEND LLAMA_DIR

# install [args...] — run install.sh into a prefix; sets RC and OUT
install() {
  OUT="$(LOCAL_CLAUDE_PREFIX="$ROOT/prefix" LOCAL_CLAUDE_STATE="$ROOT/state" MODELS_DIR="$ROOT/models" \
         LOCAL_PORT=1 "$HERE/install.sh" --no-symlink "$@" 2>&1)"
  RC=$?
}
running() { "$ROOT/prefix/current/llama-server" --version 2>/dev/null; }
check() {
  if [ "$2" = 0 ]; then echo "  ok    $1"; PASS=$((PASS+1))
  else echo "  FAIL  $1"; echo "$OUT" | tail -4 | sed 's/^/          /'; FAIL=$((FAIL+1)); fi
}

install --backend vulkan
[ "$RC" = 0 ] && [ "$(running)" = "version: 9000 (vulkan)" ]
check "a fresh install unpacks the Vulkan build and points current at it" $?

install --backend cpu
[ "$RC" = 0 ] && [ "$(running)" = "version: 9000 (cpu)" ] && [ -x "$ROOT/prefix/b9000/llama-server" ]
check "--backend cpu installs the CPU build of the same release beside it" $?

install
[ "$RC" = 0 ] && [ "$(running)" = "version: 9000 (cpu)" ]
check "a plain re-run keeps the build in use" $?

rm -rf "$ROOT/prefix" "$ROOT/state"
LOCAL_SEARCH_BACKEND=links install
[ "$RC" != 0 ] && echo "$OUT" | grep -q "unknown LOCAL_SEARCH_BACKEND=links" && [ ! -e "$ROOT/prefix/current" ]
check "an unknown LOCAL_SEARCH_BACKEND stops the installer before it installs anything" $?

echo "install: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
