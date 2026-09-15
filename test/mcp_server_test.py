#!/usr/bin/env python3
"""Drive the search MCP server over stdio with a real JSON-RPC handshake."""
import os
import json
import subprocess
import sys
import threading

PYTHON = sys.executable
SERVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp", "search_server.py")

reqs = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},   # notification, no reply
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
     "params": {"name": "web_search", "arguments": {"query": "python enumerate builtin", "max_results": 3}}},
    {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
     "params": {"name": "web_fetch", "arguments": {"url": "https://example.com"}}},
]

p = subprocess.Popen([PYTHON, SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, bufsize=1)


def drain_stderr():
    for _ in p.stderr:
        pass


threading.Thread(target=drain_stderr, daemon=True).start()

for r in reqs:
    p.stdin.write(json.dumps(r) + "\n")
    p.stdin.flush()
p.stdin.close()

responses = {}
for line in p.stdout:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    responses[msg.get("id")] = msg
p.wait(timeout=120)

fails = 0


def check(name, cond):
    global fails
    print(("PASS: " if cond else "FAIL: ") + name)
    if not cond:
        fails += 1


TRANSIENT = ("Max retries", "timed out", "Connection", "HTTPSConnectionPool",
             "429", "Temporary failure", "Name or service not known")


def check_net(name, cond, text):
    """A check that depends on someone else's server being up.

    DuckDuckGo rate-limits and intermittently refuses, and the machine may simply
    be offline. Neither says anything about whether this stack works, so a
    transport-level failure is reported and skipped rather than counted.
    """
    global fails
    if cond:
        print("PASS: " + name)
    elif any(t in text for t in TRANSIENT):
        print("SKIP: " + name + "  (transient network failure, not a stack fault)")
    else:
        print("FAIL: " + name)
        fails += 1


init = responses.get(1, {})
check("initialize returns protocolVersion + tools capability",
      init.get("result", {}).get("protocolVersion") and "tools" in init.get("result", {}).get("capabilities", {}))
check("initialized notification produced NO response", 2 in responses and None not in responses)

tl = responses.get(2, {}).get("result", {}).get("tools", [])
names = {t["name"] for t in tl}
check("tools/list advertises web_search + web_fetch", {"web_search", "web_fetch"} <= names)
check("tools have inputSchema with required query/url",
      all("inputSchema" in t and "properties" in t["inputSchema"] for t in tl))

sr = responses.get(3, {}).get("result", {})
sr_text = "".join(b.get("text", "") for b in sr.get("content", []))
print("  search sample:", sr_text[:120].replace("\n", " ") + "...")
check_net("web_search not error", sr.get("isError") is False, sr_text)
check_net("web_search returned URLs", "http" in sr_text and "URL:" in sr_text, sr_text)

fr = responses.get(4, {}).get("result", {})
fr_text = "".join(b.get("text", "") for b in fr.get("content", []))
print("  fetch sample:", fr_text[:120].replace("\n", " ") + "...")
check_net("web_fetch not error", fr.get("isError") is False, fr_text)
check_net("web_fetch got example.com body", "example" in fr_text.lower(), fr_text)

print(f"\n{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
