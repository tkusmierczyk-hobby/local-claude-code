#!/usr/bin/env bash
# pull-after.sh — wait for one file to appear, then start a model download.
#
# The link only has room for one large transfer at a time, so downloads are
# queued rather than run in parallel: two concurrent pulls each ran at half
# speed and finished later than they would have back to back.
#
# Usage:
#   setsid nohup bin/pull-after.sh <wait-for-file> <hf-repo> <file> [sha256] \
#     </dev/null >/dev/null 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WAIT_FOR="${1:?usage: pull-after.sh <wait-for-file> <repo> <file> [sha256]}"; shift
while [ ! -f "$WAIT_FOR" ]; do sleep 30; done
exec "$HERE/fetch-model.sh" "$@"
