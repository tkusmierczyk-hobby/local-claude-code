#!/usr/bin/env bash
# bench.sh — measure prefill/decode speed and tune --n-cpu-moe for this machine.
#
# --n-cpu-moe N keeps the routed experts of the first N layers in system RAM and
# leaves everything else (attention, KV cache, router, shared experts) on the GPU.
# Lower N = more expert weights in VRAM = faster, right up until VRAM overflows,
# at which point the server fails to start. The sweep walks N down and stops at
# the first value that does not fit.
#
# Then it tries a doubled --ubatch-size. Each micro-batch of prompt copies the
# RAM-resident experts to the GPU once, so a bigger one means fewer copies per
# token, but its compute buffer takes VRAM from the expert layers: it is tried
# at the smallest N that still starts, and kept only if it buys >= 10% prefill
# for <= 5% decode.
#
# "Starts" is not enough. The server keeps allocating GPU memory after start —
# CUDA graphs for each new batch shape — and a split that left 395 MiB free
# crashed mid-session with "CUDA error: out of memory" (cudaGraphInstantiate),
# while one layer more in RAM (855 MiB free) did not. So a candidate must also
# leave LOCAL_VRAM_MARGIN MiB free after its test request (default 768;
# checked where nvidia-smi exists).
#
# The sweep drives the real llama-server through `local-claude start`, not
# llama-bench: context size, slot count and KV type all consume the VRAM the
# expert layers are competing for, and llama-bench sizes its KV cache from
# -p/-n instead of from the server's --ctx-size, so it would tune against a
# smaller allocation than the one you actually run.
#
# Usage:
#   ./bench.sh                       # measure the current/default configuration
#   ./bench.sh --tune                # sweep --n-cpu-moe and the micro-batch, store both
#   ./bench.sh --tune --sweep 40,36,32,28 --ubatch 2048,4096
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

MODEL=""; CTX="${LOCAL_CTX:-${HWENV_LOCAL_CTX:-131072}}"; SWEEP=""; TUNE=0
BASE_UB="${LOCAL_UBATCH:-${HWENV_LOCAL_UBATCH:-2048}}"; UBATCHES="$BASE_UB,$(( BASE_UB * 2 ))"
MARGIN="${LOCAL_VRAM_MARGIN:-768}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tune)   TUNE=1; shift ;;
    --sweep)  SWEEP="$2"; shift 2 ;;
    --model)  MODEL="$2"; shift 2 ;;
    --ctx)    CTX="$2"; shift 2 ;;
    --ubatch) UBATCHES="$2"; shift 2 ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
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
import json, os, sys, urllib.request
base = sys.argv[1]
# ~9.5k tokens of code-shaped text: representative of what Claude Code sends.
prompt = "Summarize this module:\n" + "".join(
    f"def f{i}(x):\n    \"\"\"helper {i}\"\"\"\n    return x * {i} + {i % 7}\n\n" for i in range(340))
body = {"prompt": prompt, "n_predict": 128, "cache_prompt": False, "temperature": 0}
headers = {"Content-Type": "application/json"}
key = os.environ.get("LOCAL_API_KEY") or os.environ.get("LLAMA_API_KEY")
if key:
    headers["Authorization"] = "Bearer " + key
req = urllib.request.Request(base + "/completion", data=json.dumps(body).encode(), headers=headers)
try:
    t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
except Exception as e:
    print(f"ERR {e}", file=sys.stderr); sys.exit(1)
print(f"{t['prompt_n']:.0f} {t['prompt_per_second']:.1f} {t['predicted_per_second']:.1f}")
PY
}

IFS=',' read -ra UBS <<< "$UBATCHES"
row() { printf "%-10s %7s %10s %14s %13s %10s%s\n" "$@"; }
row "n_cpu_moe" "ubatch" "prompt tok" "prefill t/s" "decode t/s" "VRAM" ""

# try N UBATCH — start the real server with this split; sets ptok pp tg vram vfree.
try() {
  ptok="" pp="" tg="" vram="" vfree=""
  "$HERE/local-claude" stop >/dev/null 2>&1
  sleep 2
  LOCAL_NCMOE="$1" LOCAL_UBATCH="$2" LOCAL_CTX="$CTX" LOCAL_MODEL_FILE="$MODEL" \
    "$HERE/local-claude" start >/dev/null 2>&1 || return 1
  read -r ptok pp tg <<< "$(measure)"
  local used total
  read -r used total <<< "$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')"
  [[ "$used" =~ ^[0-9]+$ ]] && { vram="${used}MiB"; vfree=$(( total - used )); }
  return 0
}
# tight — the last try left less than MARGIN MiB free (unknown counts as fine)
tight() { [[ -n "$vfree" && "$vfree" -lt "$MARGIN" ]]; }

best_n=""; best_tg=0; best_pp=0; best_ub="${UBS[0]}"
IFS=',' read -ra VALUES <<< "$SWEEP"
for n in "${VALUES[@]}"; do
  [[ "$n" -lt 0 ]] && continue
  if ! try "$n" "${UBS[0]}"; then
    row "$n" "${UBS[0]}" "-" "-" "-" "-" "  did not start (out of VRAM) — stopping sweep"
    break
  fi
  if [[ -z "${tg:-}" ]]; then
    row "$n" "${UBS[0]}" "-" "-" "-" "-" "  request failed — stopping sweep"
    break
  fi
  if tight; then
    row "$n" "${UBS[0]}" "$ptok" "$pp" "$tg" "$vram" "  only $vfree MiB free (< $MARGIN) — stopping sweep"
    break
  fi
  note=""
  if LC_ALL=C awk -v a="$tg" -v b="$best_tg" 'BEGIN{exit !(a>b)}'; then
    best_tg="$tg"; best_pp="$pp"; best_n="$n"; note="  <- best"
  fi
  row "$n" "${UBS[0]}" "$ptok" "$pp" "$tg" "$vram" "$note"
  [[ "$TUNE" == "0" ]] && break   # single measurement unless sweeping
done

# Larger micro-batches, each at the smallest split that still starts.
if [[ "$TUNE" == "1" && -n "$best_n" ]]; then
  for ub in "${UBS[@]:1}"; do
    for (( n = best_n; n <= best_n + 4 && n <= LAYERS; n++ )); do
      if ! try "$n" "$ub"; then
        row "$n" "$ub" "-" "-" "-" "-" "  did not start"
        continue
      fi
      [[ -n "${tg:-}" ]] || { row "$n" "$ub" "-" "-" "-" "-" "  request failed"; break; }
      if tight; then
        row "$n" "$ub" "$ptok" "$pp" "$tg" "$vram" "  only $vfree MiB free (< $MARGIN)"
        continue
      fi
      note="  (kept the smaller micro-batch: not worth the decode)"
      if LC_ALL=C awk -v p="$pp" -v bp="$best_pp" -v t="$tg" -v bt="$best_tg" \
           'BEGIN{exit !(p >= 1.10*bp && t >= 0.95*bt)}'; then
        best_n="$n"; best_ub="$ub"; best_pp="$pp"; best_tg="$tg"; note="  <- best"
      fi
      row "$n" "$ub" "$ptok" "$pp" "$tg" "$vram" "$note"
      break
    done
  done
fi

echo
if [[ "$TUNE" == "1" && -n "$best_n" ]]; then
  # keyed by build: backends have different VRAM footprints for the same model
  build="$(basename "$(readlink -f "$LLAMA_DIR" 2>/dev/null || echo "$LLAMA_DIR")")"
  echo "$best_n" > "$STATE_DIR/ncmoe.$build"
  echo "$best_ub" > "$STATE_DIR/ubatch.$build"
  echo "$best_n" > "$STATE_DIR/ncmoe"
  echo "tuned: --n-cpu-moe $best_n --ubatch-size $best_ub ($best_pp tok/s prefill, $best_tg tok/s decode)"
  echo "       -> $STATE_DIR/ncmoe.$build, $STATE_DIR/ubatch.$build"
  # A server left running would keep serving the last configuration tried.
  "$HERE/local-claude" stop >/dev/null 2>&1
  echo "the launcher uses them from the next start"
fi
free -g | awk '/^Mem:/ {printf "ram: %d GB used / %d GB total\n", $3, $2}'
