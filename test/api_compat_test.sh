#!/usr/bin/env bash
# Tests the backend's Anthropic-compatible /v1/messages with Claude-Code-shaped
# requests: system blocks with cache_control, tools, tool_result round-trips, SSE.
# Defaults to llama-server on :8080; pass a base URL to test ollama (:11434).
#
# Usage: ./api_compat_test.sh [model] [base-url]
set -u
BASE="${2:-http://127.0.0.1:${LOCAL_PORT:-8080}}"
KEY="${LOCAL_API_KEY:-${LLAMA_API_KEY:-local}}"   # the server's key, if it has one
# llama-server serves whichever model it loaded, but the field must still be sent.
MODEL="${1:-$(curl -sf -H "x-api-key: $KEY" "$BASE/props" 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin).get("model_alias") or "local")' 2>/dev/null || echo local)}"
PASS=0; FAIL=0
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo "backend: $BASE   model: $MODEL"
check() { # name, condition
  if [ "$2" = "0" ]; then echo "PASS: $1"; PASS=$((PASS+1)); else echo "FAIL: $1"; FAIL=$((FAIL+1)); fi
}

echo "== 1. basic non-streaming message =="
# max_tokens has to be generous: these are thinking models, and a small budget is
# spent entirely inside the `thinking` block, leaving no text to assert on.
R=$(curl -sf -X POST "$BASE/v1/messages" \
  -H "content-type: application/json" -H "x-api-key: $KEY" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"'"$MODEL"'","max_tokens":4096,"messages":[{"role":"user","content":"Say OK and nothing else."}]}')
echo "$R" | python3 -c '
import json,sys
r=json.load(sys.stdin)
assert r["type"]=="message", r
blocks=[b["type"] for b in r["content"]]
print("  blocks:", blocks, "stop_reason:", r["stop_reason"], "usage:", r.get("usage"))
t="".join(b.get("text","") for b in r["content"] if b["type"]=="text")
print("  text:", t[:60].replace("\n"," "))
assert t, f"no text block (blocks={blocks}) — thinking consumed the whole budget"'
check "basic message" $?

echo "== 2. system as array of blocks + cache_control (Claude Code shape) =="
R=$(curl -sf -X POST "$BASE/v1/messages" \
  -H "content-type: application/json" -H "authorization: Bearer $KEY" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"'"$MODEL"'","max_tokens":512,
       "system":[{"type":"text","text":"You are a terse assistant.","cache_control":{"type":"ephemeral"}}],
       "metadata":{"user_id":"test-user"},
       "messages":[{"role":"user","content":[{"type":"text","text":"Reply with the word: pong"}]}]}')
echo "$R" | python3 -c 'import json,sys; r=json.load(sys.stdin); t="".join(b.get("text","") for b in r["content"] if b["type"]=="text"); assert t; print("  text:", t[:60].replace("\n"," "))'
check "system blocks + cache_control + metadata" $?

echo "== 3. tools + tool_choice auto, expect tool_use block =="
R=$(curl -sf -X POST "$BASE/v1/messages" \
  -H "content-type: application/json" -H "x-api-key: $KEY" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"'"$MODEL"'","max_tokens":512,
       "tools":[{"name":"list_files","description":"List files in a directory. Use this whenever asked about files.","input_schema":{"type":"object","properties":{"path":{"type":"string","description":"directory path"}},"required":["path"]}}],
       "messages":[{"role":"user","content":"Use the list_files tool to list files in /tmp. You must call the tool."}]}')
echo "$R" | python3 -c '
import json,sys; r=json.load(sys.stdin)
blocks=[b["type"] for b in r["content"]]
print("  blocks:", blocks, "stop_reason:", r["stop_reason"])
tu=[b for b in r["content"] if b["type"]=="tool_use"]
assert tu, "no tool_use block"
print("  tool_use:", tu[0]["name"], tu[0]["input"], "id:", tu[0]["id"][:20])'
check "tool_use emitted" $?

echo "== 4. tool_result round-trip =="
R=$(curl -sf -X POST "$BASE/v1/messages" \
  -H "content-type: application/json" -H "x-api-key: $KEY" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"'"$MODEL"'","max_tokens":512,
       "tools":[{"name":"list_files","description":"List files in a directory","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
       "messages":[
         {"role":"user","content":"List files in /tmp using the tool, then tell me how many there are."},
         {"role":"assistant","content":[{"type":"tool_use","id":"toolu_local001","name":"list_files","input":{"path":"/tmp"}}]},
         {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_local001","content":"a.txt\nb.txt\nc.txt"}]}
       ]}')
echo "$R" | python3 -c 'import json,sys; r=json.load(sys.stdin); t="".join(b.get("text","") for b in r["content"] if b["type"]=="text"); print("  text:", t[:100].replace("\n"," ")); assert "3" in t or "three" in t.lower()'
check "tool_result round-trip understood" $?

echo "== 5. streaming SSE =="
curl -sfN -X POST "$BASE/v1/messages" \
  -H "content-type: application/json" -H "x-api-key: $KEY" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"'"$MODEL"'","max_tokens":512,"stream":true,"messages":[{"role":"user","content":"Count 1 2 3"}]}' \
  | head -40 > "$TMP/sse.txt"
grep -q "event: message_start" "$TMP/sse.txt" && grep -q "content_block_delta" "$TMP/sse.txt"
check "SSE stream events" $?
grep -oE "event: [a-z_]+" "$TMP/sse.txt" | sort | uniq -c | sed 's/^/  /'

echo "== 6. count_tokens endpoint =="
# ollama 404s this and the client silently falls back to estimating; llama-server
# implements it, which makes Claude Code's context accounting exact.
R=$(curl -s -o "$TMP/ct.json" -w "%{http_code}" -X POST "$BASE/v1/messages/count_tokens" \
  -H "content-type: application/json" -H "x-api-key: $KEY" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Hello world"}]}')
echo "  http=$R body=$(head -c 200 "$TMP/ct.json")"
[ "$R" = "200" ]; check "count_tokens" $?

echo "== 7. server reports a usable context window =="
curl -sf -H "x-api-key: $KEY" "$BASE/props" 2>/dev/null | python3 -c '
import json,sys
p=json.load(sys.stdin)
n=p.get("default_generation_settings",{}).get("n_ctx",0)
print("  n_ctx:", n, " model:", (p.get("model_path") or "?").split("/")[-1])
assert n >= 32768, f"context {n} is too small for Claude Code (want >= 32768)"'
check "context window >= 32k" $?

echo; echo "RESULT: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
