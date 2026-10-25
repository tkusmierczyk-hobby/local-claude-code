#!/usr/bin/env bash
# bench.sh — measure prefill/decode speed and tune --n-cpu-moe for this machine.
#
# --n-cpu-moe N keeps the routed experts of the first N layers in system RAM and
# leaves everything else (attention, KV cache, router, shared experts) on the GPU.
# Lower N = more expert weights in VRAM = faster, right up until VRAM overflows,
# at which point the server fails to start. The sweep walks N down and stops at
# the first value that does not fit.
#
# The sweep drives the real llama-server through `local-claude start`, not
# llama-bench: context size, slot count and KV type all consume the VRAM the
# expert layers are competing for, and llama-bench sizes its KV cache from
# -p/-n instead of from the server's --ctx-size, so it would tune against a
# smaller allocation than the one you actually run.
#
# Usage:
#   ./bench.sh                       # measure the current/default configuration
#   ./bench.sh --tune                # sweep --n-cpu-moe, store the winner
#   ./bench.sh --tune --sweep 40,36,32,28
#   ./bench.sh --model ~/models/x.gguf --ctx 65536
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="${LOCAL_CLAUDE_STATE:-$HOME/.local-claude}"
# Same lookup as the launcher: the tuned value is stored per build, and it is
# only used if it lands in the file the launcher reads.
PREFIX="${LOCAL_CLAUDE_PREFIX:-$HOME/.local/share/local-claude}"
if [[ -z "${LLAMA_DIR:-}" ]]; then
  for cand in "$PREFIX/current" "$HOME/opt/llamacpp/current"; do
    [[ -x "$cand/llama-server" ]] && { LLAMA_DIR="$cand"; break; }
  done
  LLAMA_DIR="${LLAMA_DIR:-$PREFIX/current}"
fi
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
PORT="${LOCAL_PORT:-8080}"
BASE="http://127.0.0.1:$PORT"
mkdir -p "$STATE_DIR"

MODEL=""; CTX="${LOCAL_CTX:-131072}"; SWEEP=""; TUNE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tune)   TUNE=1; shift ;;
    --sweep)  SWEEP="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    --ctx)    CTX="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  for pat in 'Qwen3.6-35B-A3B*.gguf' '*35B-A3B*.gguf' '*A3B*.gguf' '*.gguf'; do
    MODEL="$(ls -t "$MODELS_DIR"/$pat 2>/dev/null | grep -v '\.part$' | grep -vi mmproj | head -1)"
    [[ -n "$MODEL" ]] && break
  done
fi
[[ -f "$MODEL" ]] || { echo "no model found (looked in $MODELS_DIR); pass --model PATH"; exit 1; }

LAYERS="$(python3 "$HERE/bin/patch-gguf.py" "$MODEL" --show 2>/dev/null \
          | awk -F= '/\.block_count /{gsub(/ /,"",$2); print $2; exit}')"
LAYERS="${LAYERS:-40}"
EXPERTS="$(python3 "$HERE/bin/patch-gguf.py" "$MODEL" --show 2>/dev/null \
          | awk -F= '/\.expert_count /{gsub(/ /,"",$2); print $2; exit}')"
EXPERTS="${EXPERTS:-0}"
[[ -z "$SWEEP" ]] && SWEEP="$LAYERS,$((LAYERS-2)),$((LAYERS-4)),$((LAYERS-6)),$((LAYERS-8))"
[[ "$EXPERTS" == "0" ]] && { SWEEP="0"; TUNE=0; echo "note: dense model (no experts) — nothing to tune"; }

echo "model:  $(basename "$MODEL")  ${LAYERS} layers, ${EXPERTS} experts"
echo "ctx:    $CTX   port: $PORT"
echo

# One request that exercises both halves: a long prompt (prefill) and a real
# generation (decode). cache_prompt=false so every run pays full prefill.
measure() {
  python3 - "$BASE" <<'PY'
import json, sys, urllib.request
base = sys.argv[1]
# ~9.5k tokens of code-shaped text: representative of what Claude Code sends.
prompt = "Summarize this module:\n" + "".join(
    f"def f{i}(x):\n    \"\"\"helper {i}\"\"\"\n    return x * {i} + {i % 7}\n\n" for i in range(340))
body = {"prompt": prompt, "n_predict": 128, "cache_prompt": False, "temperature": 0}
req = urllib.request.Request(base + "/completion", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
except Exception as e:
    print(f"ERR {e}", file=sys.stderr); sys.exit(1)
print(f"{t['prompt_n']:.0f} {t['prompt_per_second']:.1f} {t['predicted_per_second']:.1f}")
PY
}

printf "%-10s %10s %14s %13s %10s\n" "n_cpu_moe" "prompt tok" "prefill t/s" "decode t/s" "VRAM"
best_n=""; best_tg=0
IFS=',' read -ra VALUES <<< "$SWEEP"
for n in "${VALUES[@]}"; do
  [[ "$n" -lt 0 ]] && continue
  "$HERE/local-claude" stop >/dev/null 2>&1
  sleep 2
  if ! LOCAL_NCMOE="$n" LOCAL_CTX="$CTX" LOCAL_MODEL_FILE="$MODEL" \
       "$HERE/local-claude" start >/dev/null 2>&1; then
    printf "%-10s %10s %14s %13s %10s\n" "$n" "-" "-" "-" "did not start (out of VRAM) — stopping sweep"
    break
  fi
  read -r ptok pp tg <<< "$(measure)"
  vram="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -z "${tg:-}" ]]; then
    printf "%-10s %10s %14s %13s %10s\n" "$n" "-" "-" "-" "request failed — stopping sweep"
    break
  fi
  note=""
  if LC_ALL=C awk -v a="$tg" -v b="$best_tg" 'BEGIN{exit !(a>b)}'; then
    best_tg="$tg"; best_n="$n"; note="  <- best"
  fi
  printf "%-10s %10s %14s %13s %10s%s\n" "$n" "$ptok" "$pp" "$tg" "$vram" "$note"
  [[ "$TUNE" == "0" ]] && break   # single measurement unless sweeping
done

echo
if [[ "$TUNE" == "1" && -n "$best_n" ]]; then
  # keyed by build: backends have different VRAM footprints for the same model
  cache="$STATE_DIR/ncmoe.$(basename "$(readlink -f "$LLAMA_DIR" 2>/dev/null || echo "$LLAMA_DIR")")"
  echo "$best_n" > "$cache"
  echo "$best_n" > "$STATE_DIR/ncmoe"
  echo "tuned: --n-cpu-moe $best_n ($best_tg tok/s decode) -> $cache"
  echo "the launcher picks this up on the next start"
fi
free -g | awk '/^Mem:/ {printf "ram: %d GB used / %d GB total\n", $3, $2}'
