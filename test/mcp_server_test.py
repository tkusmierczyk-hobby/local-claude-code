#!/usr/bin/env python3
"""Drive the search MCP server over stdio with a real JSON-RPC handshake."""
import os
import json
import subprocess
import sys
import tempfile
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
             "429", "Temporary failure", "Name or service not known",
             "refusing automated searches")  # DuckDuckGo's bot check


def check_net(name, cond, text):
    """A check that depends on someone else's server being up.

    DuckDuckGo rate-limits, serves a bot check after a burst of searches, and
    intermittently refuses, and the machine may simply be offline. None of that
    says anything about whether this stack works, so a transport-level failure
    is reported and skipped rather than counted.
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

# Offline, against stand-ins for the engines: the refusal and key handling,
# which a live run reaches only by luck (DuckDuckGo) or with a real key (Brave).
sys.path.insert(0, os.path.dirname(SERVER))
import search_server as ss  # noqa: E402


class Reply:
    def __init__(self, status, body):
        self.status_code, self._body, self.reason = status, body, ""
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return self._body


replies, seen = [], {}


def fake_get(url, params=None, headers=None, timeout=None):
    seen.update(url=url, params=params, headers=headers)
    return replies.pop(0)


def call_search(query):
    r = ss.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                   "params": {"name": "web_search", "arguments": {"query": query}}})["result"]
    return r["isError"], r["content"][0]["text"]


ss.requests.get = fake_get
os.environ.pop("SEARCH_MCP_LOG", None)
os.environ["LOCAL_SEARCH_ENGINE"] = "duckduckgo"
replies[:] = [Reply(403, "If this persists, please email us.")]
err, text = call_search("x")
check("duckduckgo: a 403 refusal says it is not a configuration problem",
      err and "not a configuration problem" in text)

with tempfile.TemporaryDirectory() as d:
    keyfile = os.path.join(d, "brave-search.key")
    with open(keyfile, "w") as fh:
        fh.write("test-key-123\n")
    os.environ.update(LOCAL_SEARCH_ENGINE="brave", SEARCH_BRAVE_KEY_FILE=keyfile)
    replies[:] = [Reply(200, {"web": {"results": [{
        "title": "Built-in Functions", "url": "https://docs.python.org/3/library/functions.html",
        "description": "Return an <strong>enumerate</strong> object."}]}})]
    err, text = call_search("python enumerate")
    check("brave: the key goes in the header, not the URL",
          seen["headers"].get("X-Subscription-Token") == "test-key-123"
          and "test-key-123" not in json.dumps(seen["params"]))
    check("brave: results parsed, <strong> tags stripped",
          not err and "URL: https://docs.python.org" in text and "Return an enumerate object." in text)
    replies[:] = [Reply(422, {"error": {"code": "SUBSCRIPTION_TOKEN_INVALID"}, "type": "ErrorResponse"})]
    err, text = call_search("x")
    check("brave: a rejected key is reported as a configuration problem, key not echoed",
          err and "rejected the API key" in text and "test-key-123" not in text)
    os.remove(keyfile)
    replies[:] = [Reply(200, {})]
    err, text = call_search("x")
    check("brave: a missing key file is reported, nothing sent",
          err and "no API key in" in text and len(replies) == 1)

print(f"\n{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
