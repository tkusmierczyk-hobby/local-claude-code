#!/usr/bin/env bash
# fetch-model.sh — resumable HuggingFace GGUF downloader for the local-claude stack.
#
# Built for slow links: the connection here is ~0.5 MB/s, so a 22 GB model is a
# ~13 h download. curl is restarted with -C - until the file is complete, so
# suspends, drops and reboots cost only the current chunk.
#
# Usage: fetch-model.sh <repo> <file> [sha256]
#   repo   HF repo id, e.g. unsloth/Qwen3.6-35B-A3B-GGUF
#   file   file in that repo, e.g. Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
#   sha256 optional expected hash; verified once the size matches
#
# Env: MODELS_DIR (default ~/models), FETCH_LOG (default $MODELS_DIR/<file>.log),
#      HF_ENDPOINT (default https://huggingface.co: a mirror, or a test stand-in),
#      FETCH_PAUSE (seconds between attempts, default 5)
set -uo pipefail

REPO="${1:?usage: fetch-model.sh <repo> <file> [sha256]}"
FILE="${2:?usage: fetch-model.sh <repo> <file> [sha256]}"
WANT_SHA="${3:-}"

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
HF="${HF_ENDPOINT:-https://huggingface.co}"; HF="${HF%/}"
URL="${HF}/${REPO}/resolve/main/${FILE}"
DEST="${MODELS_DIR}/${FILE}"
PART="${DEST}.part"
PAUSE="${FETCH_PAUSE:-5}"
LOG="${FETCH_LOG:-${DEST}.log}"

mkdir -p "$MODELS_DIR"
exec >>"$LOG" 2>&1

# Killing this script must take its curl child with it, otherwise an orphaned
# curl keeps writing to the .part file and competing for the link.
CURL_PID=""
trap '[ -n "$CURL_PID" ] && kill "$CURL_PID" 2>/dev/null; exit 143' INT TERM
echo "=== $(date -Is) fetch ${REPO}/${FILE} -> ${DEST} ==="

# Expected size from the HF API, so we know when we are actually done.
total="$(curl -sSL "${HF}/api/models/${REPO}/tree/main?recursive=1" \
  | python3 -c "
import sys,json
f=sys.argv[1]
for e in json.load(sys.stdin):
    if e.get('path')==f: print(e.get('size',0)); break
else: print(0)
" "$FILE")"
echo "expected size: ${total:-unknown} bytes"

if [ -f "$DEST" ] && [ "$(stat -c%s "$DEST")" = "$total" ]; then
  echo "already complete: $DEST"; exit 0
fi

attempt=0
while :; do
  attempt=$((attempt + 1))
  have=0; [ -f "$PART" ] && have="$(stat -c%s "$PART")"
  if [ "$total" != "0" ] && [ "$have" -ge "$total" ]; then break; fi
  pct=0; [ "$total" != "0" ] && pct=$((have * 100 / total))
  echo "--- $(date -Is) attempt ${attempt}: ${have}/${total} bytes (${pct}%)"
  # Deliberately NO --retry here. curl's internal retry resumes from the offset it
  # computed when the process started, not from where the file actually got to, so
  # with -C - every internal retry re-downloads the same window and throws it away
  # (this quietly burned ~8 GB and eight hours before it was spotted). The loop
  # above is the retry: it re-stats the file each pass, so -C - always re-anchors
  # on real progress. --max-time is short for the same reason — it bounds how much
  # a single connection can redo. --speed-limit kills sockets that hang while open.
  # -f keeps HTTP errors out of the file: without it, an error page answering the
  # first request became the start of the .part, and every chunk landed after it.
  curl -fsSL -C - -o "$PART" --connect-timeout 30 --max-time 1800 \
       --speed-limit 10000 --speed-time 60 \
       -w "  chunk: %{size_download} B @ %{speed_download} B/s\n" "$URL" &
  CURL_PID=$!
  wait "$CURL_PID"
  CURL_PID=""
  sleep "$PAUSE"
done

echo "download complete: $(stat -c%s "$PART") bytes"
if [ -n "$WANT_SHA" ]; then
  echo "verifying sha256 (a few minutes for a 20 GB file)..."
  got="$(sha256sum "$PART" | cut -d' ' -f1)"
  if [ "$got" != "$WANT_SHA" ]; then
    echo "SHA MISMATCH: got $got want $WANT_SHA — leaving $PART in place for inspection"; exit 1
  fi
  echo "sha256 OK"
fi
mv -f "$PART" "$DEST"
echo "=== $(date -Is) done: $DEST ==="
