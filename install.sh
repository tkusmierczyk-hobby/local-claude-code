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
#   ./install.sh --no-symlink    # skip ~/.local/bin/local-claude
#
# Claude Code itself is assumed to be installed already.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${LOCAL_CLAUDE_PREFIX:-$HOME/.local/share/local-claude}"
STATE_DIR="${LOCAL_CLAUDE_STATE:-$HOME/.local-claude}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
BIN_DIR="$HOME/.local/bin"
BACKEND_OVERRIDE=""; BUILD=""; DO_SYMLINK=1

while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND_OVERRIDE="$2"; shift 2 ;;
    --build)   BUILD="$2"; shift 2 ;;
    --no-symlink) DO_SYMLINK=0; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

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

# Careful: llama.cpp publishes version tags (v0.4.x) that carry *no* binaries
# alongside build tags (bNNNNN) that do, and /releases/latest returns the former.
# So walk recent releases and take the newest one that actually has our asset,
# using the URL the API reports rather than constructing a filename.
info "looking for the newest release with a $ASSET build..."
read -r BUILD TARBALL URL <<<"$(
  curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30" \
  | ASSET="$ASSET" WANT="${BUILD:-}" python3 -c '
import json, os, sys
asset, want = os.environ["ASSET"], os.environ.get("WANT") or ""
for rel in json.load(sys.stdin):
    if want and rel["tag_name"] != want:
        continue
    for a in rel.get("assets", []):
        n = a["name"]
        if n.endswith(".tar.gz") and asset in n:
            print(rel["tag_name"], n, a["browser_download_url"])
            sys.exit(0)
sys.exit(1)
')" || die "no llama.cpp release with a $ASSET build in the last 30 releases${BUILD:+ (tag $BUILD)}; pass --build <tag>"
info "build     $BUILD"
info "asset     $TARBALL"

# ---- 3. install it -----------------------------------------------------------
say "3/6  Installing llama.cpp (user-space, no root)"
DEST="$PREFIX/$BUILD"
if [ -x "$DEST/llama-server" ]; then
  info "already installed at $DEST"
else
  mkdir -p "$PREFIX/dl" "$DEST"
  if [ ! -s "$PREFIX/dl/$TARBALL" ]; then
    info "downloading (resumable)..."
    curl -fL -C - --retry 3 --progress-bar -o "$PREFIX/dl/$TARBALL" "$URL" \
      || die "download failed: $URL"
  fi
  tar xzf "$PREFIX/dl/$TARBALL" -C "$DEST" --strip-components=1 2>/dev/null \
    || tar xzf "$PREFIX/dl/$TARBALL" -C "$DEST"
  # Release layouts have varied; find the binaries wherever they landed.
  if [ ! -x "$DEST/llama-server" ]; then
    found="$(find "$DEST" -name llama-server -type f | head -1)"
    [ -n "$found" ] || die "llama-server not found inside $TARBALL"
    cp -a "$(dirname "$found")"/* "$DEST"/ 2>/dev/null || true
  fi
  chmod +x "$DEST"/llama-* 2>/dev/null || true
  info "installed to $DEST"
fi
ln -sfn "$DEST" "$PREFIX/current"
info "current -> $DEST"

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
      "args": ["$HERE/mcp/search_server.py"],
      "env": {
        "SEARCH_MCP_LOG": "$STATE_DIR/search-mcp.log"
      }
    }
  }
}
JSON
info "mcp       $STATE_DIR/search-mcp.json"
if ! "$PY" -c 'import requests, bs4' >/dev/null 2>&1; then
  info "note: web search needs 'requests' and 'beautifulsoup4' — pip install them, or set LOCAL_SEARCH=0"
fi

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

if [ "$HW_BACKEND" = vulkan ] && [ "$HW_GPU_NAME" != "${HW_GPU_NAME#NVIDIA}" ]; then
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
