#!/usr/bin/env bash
# install.sh — set up the local Claude Code stack on this machine.
#
# Installs nothing system-wide and needs no root: llama.cpp goes to
# ~/.local/share/local-claude/llamacpp, state to ~/.local-claude, models to
# ~/models, and the launcher is symlinked into ~/.local/bin.
#
#   ./install.sh                 # detect hardware, install, verify
#   ./install.sh --backend cpu   # force a backend (vulkan|cpu|metal)
#   ./install.sh --build BUILD   # pin a llama.cpp build instead of the latest
#   ./install.sh --update        # move to the newest llama.cpp release
#   ./install.sh --no-symlink    # skip ~/.local/bin/local-claude
#
# Claude Code itself is assumed to be installed already.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${LOCAL_CLAUDE_PREFIX:-$HOME/.local/share/local-claude}"
STATE_DIR="${LOCAL_CLAUDE_STATE:-$HOME/.local-claude}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
BIN_DIR="$HOME/.local/bin"
BACKEND_OVERRIDE=""; BUILD=""; DO_SYMLINK=1; UPDATE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND_OVERRIDE="$2"; shift 2 ;;
    --build)   BUILD="$2"; shift 2 ;;
    --no-symlink) DO_SYMLINK=0; shift ;;
    --update)  UPDATE=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
sha256_of() { { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; } | cut -d' ' -f1; }

# The search server sessions get by default: "requests" (search_server.py,
# needs requests+bs4) or "lynx" (search_server_lynx.py, needs lynx on PATH).
# Checked here, before anything is installed.
SEARCH_BACKEND="${LOCAL_SEARCH_BACKEND:-requests}"
case "$SEARCH_BACKEND" in
  requests) MCP_SCRIPT="$HERE/mcp/search_server.py" ;;
  lynx)     MCP_SCRIPT="$HERE/mcp/search_server_lynx.py" ;;
  *)        die "unknown LOCAL_SEARCH_BACKEND=$SEARCH_BACKEND (use requests or lynx)" ;;
esac

# fetch URL FILE SIZE SHA256 — a download that survives a slow, flaky link. It
# resumes into FILE.part and renames only once size and checksum match, so an
# interrupted run neither leaves a truncated file that looks complete nor starts
# over. A FILE already present is checked the same way: older installers could
# leave a truncated one. No curl --retry: with -C - it resumes from a stale
# offset and re-downloads the same bytes (bin/fetch-model.sh); this loop is the retry,
# and --speed-limit gives up on a connection that stalls while still open.
fetch() {
  local url="$1" file="$2" size="$3" sha="$4" part="$2.part" have n=0
  [ "$size" -gt 0 ] || die "the release lists no size for $(basename "$file")"
  [ -f "$file" ] && mv -f "$file" "$part"
  while [ ! -f "$file" ]; do
    have=0; [ -f "$part" ] && have=$(( $(wc -c < "$part") ))
    [ "$have" -gt "$size" ] && { rm -f "$part"; have=0; }
    if [ "$have" -lt "$size" ]; then
      n=$((n + 1))
      [ "$n" -le 10 ] || die "download incomplete after 10 attempts; re-run ./install.sh to resume"
      info "downloading $(basename "$file"): $(( have / 1048576 )) of $(( size / 1048576 )) MB so far"
      curl -fL -C - --connect-timeout 30 --speed-limit 10000 --speed-time 60 \
           --progress-bar -o "$part" "$url" || sleep 5
      continue
    fi
    if [ "$sha" != "-" ] && [ "$(sha256_of "$part")" != "$sha" ]; then
      rm -f "$part"
      die "checksum mismatch for $(basename "$file"); deleted it, re-run ./install.sh"
    fi
    mv -f "$part" "$file"
    if [ "$sha" = "-" ]; then info "downloaded $(basename "$file") (the release publishes no checksum)"
    else info "verified  $(basename "$file") (sha256 ${sha:0:12}…)"; fi
  done
}

# ---- 1. what are we working with --------------------------------------------
say "1/6  Inspecting this machine"
eval "$("$HERE/bin/detect-hardware.sh")"
[ -n "$BACKEND_OVERRIDE" ] && HW_BACKEND="$BACKEND_OVERRIDE"
info "os/arch   $HW_OS/$HW_ARCH"
info "cpu       $HW_CORES physical cores (using $HW_THREADS threads)"
info "ram       $(( HW_RAM_MB / 1024 )) GB — about $(( REC_USABLE_RAM_MB / 1024 )) GB usable for a model"
info "gpu       $HW_GPU_NAME${HW_VRAM_MB:+ (${HW_VRAM_MB} MB)}"
info "backend   $HW_BACKEND"
[ "$HW_OS" = unknown ] && die "unsupported OS: $(uname -s)"

for cmd in curl python3 awk sed grep tar; do
  command -v "$cmd" >/dev/null 2>&1 || die "missing required command: $cmd"
done
command -v claude >/dev/null 2>&1 \
  || info "note: the 'claude' CLI is not on PATH — install Claude Code before running local-claude"

# ---- 2. pick the llama.cpp build --------------------------------------------
say "2/6  Selecting a llama.cpp build"
case "$HW_OS/$HW_ARCH/$HW_BACKEND" in
  linux/x64/vulkan)  ASSET="bin-ubuntu-vulkan-x64" ;;
  linux/x64/cpu)     ASSET="bin-ubuntu-x64" ;;
  linux/arm64/*)     ASSET="bin-ubuntu-arm64" ;;
  macos/arm64/*)     ASSET="bin-macos-arm64" ;;
  macos/x64/*)       ASSET="bin-macos-x64" ;;
  *) die "no prebuilt llama.cpp for $HW_OS/$HW_ARCH/$HW_BACKEND — build from source and set LLAMA_DIR" ;;
esac
# Builds are installed by tag. The CPU-only download gets a -cpu suffix where
# a GPU one exists for the same tag: sharing a folder, asking for the other
# backend found the first one "already installed" and kept it.
SUFFIX=""; [ "$ASSET" = bin-ubuntu-x64 ] && SUFFIX="-cpu"

# Careful: llama.cpp publishes version tags (v0.4.x) that carry *no* binaries
# alongside build tags (bNNNNN) that do, and /releases/latest returns the former.
# So walk recent releases and take the newest one that actually has our asset,
# using the URL the API reports rather than constructing a filename. A pinned
# build is fetched by its tag, however old; if it is already installed, the
# network is not needed at all.
#
# A build that already works stays in use unless asked otherwise: re-running the
# installer used to repoint `current` at the newest release, silently replacing a
# hand-built CUDA backend with Vulkan, and leaving the per-build tuning behind.
# --update, --build and --backend each ask for a change.
TARBALL="" URL="" SIZE=0 SHA="" KEEP=0 CURRENT=""
if [ -x "$PREFIX/current/llama-server" ] \
   && LD_LIBRARY_PATH="$PREFIX/current" "$PREFIX/current/llama-server" --version >/dev/null 2>&1; then
  CURRENT="$(basename "$(readlink -f "$PREFIX/current")")"
fi
if [ -n "$CURRENT" ] && [ -z "$BUILD$BACKEND_OVERRIDE" ] && [ "$UPDATE" = 0 ]; then
  KEEP=1; BUILD="$CURRENT"
  info "build     $CURRENT (in use; ./install.sh --update moves to the newest release)"
elif [ -n "$BUILD" ] && [ -x "$PREFIX/$BUILD$SUFFIX/llama-server" ]; then
  info "build     $BUILD (already installed)"
else
  if [ -n "$BUILD" ]; then
    API="https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$BUILD"
  else
    API="https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"
  fi
  info "looking for ${BUILD:-the newest release} with a $ASSET build..."
  # Two steps, so that each failure is caught: a command failing inside
  # `read <<< "$(...)"` goes unnoticed, and the script used to carry on with an
  # empty build name until tar failed on a directory.
  releases="$(curl -fsSL --connect-timeout 20 --max-time 120 "$API")" \
    || die "could not reach the GitHub API${BUILD:+ for release $BUILD}: offline, rate-limited (60 requests/hour)${BUILD:+, or no such tag}"
  found="$(printf '%s' "$releases" | ASSET="$ASSET" python3 -c '
import json, os, sys
data = json.load(sys.stdin)
for rel in (data if isinstance(data, list) else [data]):
    for a in rel.get("assets", []):
        n = a["name"]
        if n.endswith(".tar.gz") and os.environ["ASSET"] in n:
            d = a.get("digest") or ""
            sha = d.split(":", 1)[1] if d.startswith("sha256:") else "-"
            print(rel["tag_name"], n, a["browser_download_url"], a.get("size", 0), sha)
            sys.exit(0)
sys.exit(1)
')" || die "no llama.cpp release ${BUILD:+$BUILD }has a $ASSET build${BUILD:-; pass --build <tag>}"
  read -r BUILD TARBALL URL SIZE SHA <<<"$found"
  info "build     $BUILD"
  info "asset     $TARBALL"
fi

# ---- 3. install it -----------------------------------------------------------
say "3/6  Installing llama.cpp (user-space, no root)"
DEST="$PREFIX/$BUILD$SUFFIX"
if [ "$KEEP" = 1 ]; then
  DEST="$(readlink -f "$PREFIX/current")"
  info "keeping $DEST"
elif [ -x "$DEST/llama-server" ]; then
  info "already installed at $DEST"
else
  mkdir -p "$PREFIX/dl" "$DEST"
  fetch "$URL" "$PREFIX/dl/$TARBALL" "$SIZE" "$SHA"
  # Unpack into a scratch directory, then take the directory that holds
  # llama-server. Release layouts have varied (a top-level folder, build/bin/,
  # or none), and guessing with --strip-components can strip the binaries away
  # while tar still reports success.
  unpack="$(mktemp -d "$PREFIX/unpack.XXXXXX")"
  tar xzf "$PREFIX/dl/$TARBALL" -C "$unpack" \
    || { rm -rf "${unpack:?}"; die "could not unpack $TARBALL (disk full?)"; }
  found="$(find "$unpack" -name llama-server -type f | head -1)"
  [ -n "$found" ] || { rm -rf "${unpack:?}"; die "llama-server not found inside $TARBALL"; }
  cp -a "$(dirname "$found")"/. "$DEST"/
  rm -rf "${unpack:?}"
  chmod +x "$DEST"/llama-* 2>/dev/null || true
  info "installed to $DEST"
fi
if [ "$KEEP" = 0 ]; then
  ln -sfn "$DEST" "$PREFIX/current"
  info "current -> $DEST"
  if [ -n "$CURRENT" ] && [ "$CURRENT" != "$(basename "$DEST")" ]; then
    info "note: switched from $CURRENT. Its tuning does not carry over: run"
    info "      local-claude bench --tune. To go back: ln -sfn $PREFIX/$CURRENT $PREFIX/current"
  fi
fi

VER="$(LD_LIBRARY_PATH="$PREFIX/current" "$PREFIX/current/llama-server" --version 2>&1 | head -1)"
info "$VER"

# ---- 4. state, model dir, MCP config ----------------------------------------
say "4/6  Preparing state"
mkdir -p "$STATE_DIR/templates" "$MODELS_DIR"
info "state     $STATE_DIR"
info "models    $MODELS_DIR"

# The MCP config must name absolute paths, and the right python for *this*
# machine, so it is generated here rather than shipped.
PY="$(command -v python3)"
cat > "$STATE_DIR/search-mcp.json" <<JSON
{
  "mcpServers": {
    "search": {
      "command": "$PY",
      "args": ["$MCP_SCRIPT"],
      "env": {
        "SEARCH_MCP_LOG": "$STATE_DIR/search-mcp.log"
      }
    }
  }
}
JSON
info "mcp       $STATE_DIR/search-mcp.json"
case "$SEARCH_BACKEND" in
  requests)
    if ! "$PY" -c 'import requests, bs4' >/dev/null 2>&1; then
      info "note: web search needs 'requests' and 'beautifulsoup4' — pip install them, or set LOCAL_SEARCH=0"
    fi ;;
  lynx)
    if ! command -v lynx >/dev/null 2>&1; then
      info "note: lynx search needs lynx on PATH — apt install lynx, or set LOCAL_SEARCH_BACKEND=requests"
    fi ;;
esac

cat > "$STATE_DIR/hardware.env" <<ENVEOF
# written by install.sh — the launcher reads these as defaults
LOCAL_CTX=$REC_CTX
LOCAL_UBATCH=$REC_UBATCH
LOCAL_THREADS=$HW_THREADS
LOCAL_BACKEND_KIND=$HW_BACKEND
ENVEOF
info "defaults  ctx=$REC_CTX ubatch=$REC_UBATCH threads=$HW_THREADS"

# ---- 5. put the launcher on PATH --------------------------------------------
say "5/6  Linking the launcher"
if [ "$DO_SYMLINK" = 1 ]; then
  mkdir -p "$BIN_DIR"
  ln -sfn "$HERE/local-claude" "$BIN_DIR/local-claude"
  info "$BIN_DIR/local-claude -> $HERE/local-claude"
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) info "note: $BIN_DIR is not on your PATH — add it to your shell profile" ;;
  esac
else
  info "skipped (--no-symlink); run it as $HERE/local-claude"
fi

# ---- 6. verify ---------------------------------------------------------------
say "6/6  Verifying"
"$HERE/test/preflight.sh" || die "preflight failed — see the messages above"

if [ "$HW_BACKEND" = vulkan ] && [ "$HW_GPU_NAME" != "${HW_GPU_NAME#NVIDIA}" ] \
   && [ "${BUILD%-cuda}" = "$BUILD" ]; then
  say "Optional: a CUDA build is faster"
  info "Vulkan needs nothing installed, which is why it is the default. On the"
  info "reference machine a source-built CUDA backend was worth ~20% decode and"
  info "~50% prefill. If you can get nvcc:  ./bin/build-cuda.sh   (see docs/INSTALL.md)"
fi

say "Done. Next: choose a model."
case "$REC_TIER" in
  full)
    cat <<'NEXT'
  This machine can run the full-size recommendation (22.4 GB download):

    local-claude pull unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
      707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450
NEXT
    ;;
  reduced)
    cat <<'NEXT'
  RAM is tight for the full quantisation; use the smaller one (16.9 GB):

    local-claude pull unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
NEXT
    ;;
  small)
    cat <<'NEXT'
  Not enough RAM for a 35B; a smaller MoE fits (14.3 GB):

    local-claude pull unsloth/gemma-4-26B-A4B-it-qat-GGUF gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
NEXT
    ;;
  *)
    cat <<'NEXT'
  This machine has little spare RAM. Start with the small test model (1.3 GB):

    local-claude pull unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf \
      aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223
NEXT
    ;;
esac
cat <<'NEXT'

  Large downloads are slow and resumable — run them detached:

    setsid nohup local-claude pull <repo> <file> [sha256] </dev/null >/dev/null 2>&1 &
    local-claude models          # progress

  Then:

    local-claude bench --tune    # find the best expert split for this GPU
    local-claude                 # start coding

  See docs/MODELS.md for the full model matrix and docs/TUNING.md for the knobs.
NEXT
