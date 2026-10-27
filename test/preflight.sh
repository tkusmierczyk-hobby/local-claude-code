#!/usr/bin/env bash
# preflight.sh — verify the installation before any model is involved.
#
# Everything here is cheap and needs no GGUF, so it can run right after
# install.sh and again whenever something looks wrong.
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${LOCAL_CLAUDE_PREFIX:-$HOME/.local/share/local-claude}"
STATE_DIR="${LOCAL_CLAUDE_STATE:-$HOME/.local-claude}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
LLAMA_DIR="${LLAMA_DIR:-$PREFIX/current}"
PASS=0; FAIL=0; WARN=0

ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; WARN=$((WARN+1)); }

echo "preflight: $HERE"
eval "$("$HERE/bin/detect-hardware.sh")"

# ---- binaries ----------------------------------------------------------------
if [ -x "$LLAMA_DIR/llama-server" ]; then
  v="$(LD_LIBRARY_PATH="$LLAMA_DIR" "$LLAMA_DIR/llama-server" --version 2>&1 | head -1)"
  ok "llama-server runs — $v"
else
  bad "llama-server not found at $LLAMA_DIR — run ./install.sh"
fi

if [ -x "$LLAMA_DIR/llama-bench" ]; then ok "llama-bench present"; else warn "llama-bench missing (bench --tune needs it)"; fi

# ---- the flag the whole stack depends on -------------------------------------
if [ -x "$LLAMA_DIR/llama-server" ]; then
  if LD_LIBRARY_PATH="$LLAMA_DIR" "$LLAMA_DIR/llama-server" --help 2>&1 | grep -q -- "--n-cpu-moe"; then
    ok "this build supports --n-cpu-moe"
  else
    bad "this llama.cpp build has no --n-cpu-moe — it is too old for this stack"
  fi
fi

# ---- compute device ----------------------------------------------------------
if [ -x "$LLAMA_DIR/llama-server" ]; then
  devs="$(LD_LIBRARY_PATH="$LLAMA_DIR" "$LLAMA_DIR/llama-server" --list-devices 2>/dev/null | grep -E '^\s+[A-Za-z]+[0-9]+:' || true)"
  if [ -n "$devs" ]; then
    echo "$devs" | sed 's/^/        /'
    # ranked as the launcher's detect_device does
    disc="$(echo "$devs" | awk -F: '{ s = 0 } /NVIDIA/ { s = 3 } /Radeon RX|Arc\([Tt][Mm]\) [AB][0-9][0-9][0-9]/ { s = 2 }
                                    /Apple/ { s = 1 } s > best { best = s; d = $1 } END { gsub(/ /, "", d); print d }')"
    if [ -n "$disc" ]; then
      ok "discrete GPU selected: $disc"
    else
      warn "no discrete GPU among the devices — it will run on CPU, which is slow for a 35B"
    fi
  else
    warn "no GPU devices reported; CPU-only inference"
  fi
fi

# ---- memory and disk ---------------------------------------------------------
[ "$HW_RAM_MB" -ge 16000 ] && ok "ram ${HW_RAM_MB} MB (tier: $REC_TIER)" \
  || warn "ram ${HW_RAM_MB} MB is small — expect to run a reduced model (tier: $REC_TIER)"

mkdir -p "$MODELS_DIR" 2>/dev/null
free_gb=$(df -Pk "$MODELS_DIR" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')
[ "${free_gb:-0}" -ge 25 ] && ok "disk ${free_gb} GB free in $MODELS_DIR" \
  || warn "only ${free_gb:-?} GB free in $MODELS_DIR — a full model needs ~23 GB"

# ---- state -------------------------------------------------------------------
[ -d "$STATE_DIR" ] && ok "state dir $STATE_DIR" || bad "missing state dir $STATE_DIR — run ./install.sh"
[ -s "$STATE_DIR/search-mcp.json" ] && ok "web-search MCP config generated" \
  || warn "no $STATE_DIR/search-mcp.json — run ./install.sh, or set LOCAL_SEARCH=0"

# ---- host tooling ------------------------------------------------------------
command -v claude >/dev/null 2>&1 && ok "claude CLI: $(claude --version 2>&1 | head -1)" \
  || bad "the 'claude' CLI is not on PATH — install Claude Code first"
python3 -c 'import requests, bs4' >/dev/null 2>&1 && ok "python deps for web search present" \
  || warn "python 'requests'/'beautifulsoup4' missing — web search will not load"

# ---- port --------------------------------------------------------------------
port="${LOCAL_PORT:-8080}"
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$port\$"; then
  if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    ok "port $port already serving a local-claude backend"
  else
    warn "port $port is in use by something else — set LOCAL_PORT"
  fi
else
  ok "port $port free"
fi

echo
printf '  %d ok, %d warnings, %d failures\n' "$PASS" "$WARN" "$FAIL"
[ "$FAIL" -eq 0 ]
