#!/usr/bin/env python3
"""Drive the search MCP server over stdio with a real JSON-RPC handshake."""
import os
import json
import subprocess
import sys
import tempfile
import threading
import time

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
             "could not resolve host", "unavailable right now")  # incl. DuckDuckGo's bot check


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
check_net("web_fetch got example.com body", "example domain" in fr_text.lower(), fr_text)
# Offline, against stand-ins for every site and a scratch state folder: pacing,
# fallback and key handling, which a live run reaches only by luck or with a key.
sys.path.insert(0, os.path.dirname(SERVER))
import search_server as ss  # noqa: E402


class Reply:
    def __init__(self, status, body, url=ss.DuckDuckGo.URL, location=None):
        self.status_code, self._body, self.reason, self.url = status, body, "", url
        self.text = body if isinstance(body, str) else json.dumps(body)
        self.is_redirect, self.headers = location is not None, {"location": location}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ss.requests.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        pass


class Site:
    """Hands out queued replies and records each request."""

    def __init__(self):
        self.replies, self.sent = [], []

    def __call__(self, url, params=None, data=None, headers=None, **kw):
        self.sent.append({"url": url, "params": params, "headers": headers})
        return self.replies.pop(0)


ddg, brave = Site(), Site()
ss.DDG.session.post = ss.DDG.session.get = ddg
ss.requests.get = brave  # Brave, SearXNG and web_fetch use plain requests


def call(tool, **args):
    r = ss.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                   "params": {"name": tool, "arguments": args}})["result"]
    return r["isError"], r["content"][0]["text"]


RESULTS = ''.join([
    '<tr><td><a class="result-link" href="https://example.com/a">A</a></td></tr>',
    '<tr><td class="result-snippet">about a</td></tr>',
    '<tr class="result-sponsored"><td><a class="result-link" href="https://ad.example/">Ad</a>',
    '<td class="result-snippet">buy</td></tr>',
    '<tr><td><a class="result-link" href="https://example.com/b">B</a></td></tr>',
    '<tr><td><a class="result-link" href="https://example.com/c">C</a></td></tr>',
    '<tr><td class="result-snippet">about c</td></tr>'])
BOT_CHECK = Reply(202, '<form id="challenge-form">')
check("duckduckgo: results keep their own snippets, ads dropped",
      [(r["url"], r["snippet"]) for r in ss.parse_ddg_lite(RESULTS)] == [
          ("https://example.com/a", "about a"), ("https://example.com/b", ""),
          ("https://example.com/c", "about c")])

with tempfile.TemporaryDirectory() as d:
    for var in ("SEARCH_BACKENDS", "SEARXNG_URL", "SEARCH_MCP_LOG"):
        os.environ.pop(var, None)
    keyfile = os.path.join(d, "brave-search.key")
    os.environ.update(LOCAL_CLAUDE_STATE=d, SEARCH_BRAVE_KEY_FILE=keyfile, SEARCH_CACHE_TTL="0")

    def pacing(**change):
        path = os.path.join(d, "ddg-pacing.json")
        with open(path) as fh:
            st = json.load(fh)
        st.update(change)
        with open(path, "w") as fh:
            json.dump(st, fh)
        return st

    ss.DDG.min_interval = 0.5
    ddg.replies[:] = [Reply(200, RESULTS)]
    err, text = call("web_search", query="a")
    check("duckduckgo: searched first", not err and "(via ddg)" in text)
    pacing(last=time.time())  # as if another session had just searched
    ddg.replies[:] = [Reply(200, RESULTS)]
    t0 = time.time()
    call("web_search", query="b")
    check("duckduckgo: waits its turn after another session's search", time.time() - t0 >= 0.45)

    ddg.replies[:] = [BOT_CHECK]
    err, text = call("web_search", query="c")
    check("duckduckgo: a bot check is reported as not a configuration problem",
          err and "not a configuration problem" in text and "tell the user" not in text)
    n = len(ddg.sent)
    err, text = call("web_search", query="d")
    check("duckduckgo: nothing is sent during the pause after a bot check",
          err and len(ddg.sent) == n)
    first = pacing(until=0)["strikes"]  # the pause is over, but the bot check is still there
    ddg.replies[:] = [BOT_CHECK]
    call("web_search", query="e")
    st = pacing()
    check("duckduckgo: a bot check right after a pause doubles the next one",
          first == 1 and st["strikes"] == 2 and 590 < st["until"] - time.time() <= 600)
    pacing(until=0)
    ddg.replies[:] = [Reply(200, RESULTS)]
    call("web_search", query="f")
    check("duckduckgo: a result ends the doubling", pacing()["strikes"] == 0)

    with open(keyfile, "w") as fh:
        fh.write("test-key-123\n")
    ddg.replies[:] = [BOT_CHECK]
    brave.replies[:] = [Reply(200, {"web": {"results": [{
        "title": "Built-in Functions", "url": "https://docs.python.org/3/library/functions.html",
        "description": "Return an <strong>enumerate</strong> object."}]}})]
    err, text = call("web_search", query="python enumerate")
    check("fallback: while DuckDuckGo refuses, Brave answers",
          not err and not ddg.replies and "(via brave)" in text and "Return an enumerate object." in text)
    check("brave: the key goes in the header, not the URL",
          brave.sent[-1]["headers"]["X-Subscription-Token"] == "test-key-123"
          and "test-key-123" not in json.dumps(brave.sent[-1]["params"]))
    brave.replies[:] = [Reply(422, {"error": {"code": "SUBSCRIPTION_TOKEN_INVALID"}})]
    err, text = call("web_search", query="g")
    check("brave: a rejected key is reported as a configuration problem, key not echoed",
          err and "configuration problem: tell the user" in text and "test-key-123" not in text)
    brave.replies[:] = [Reply(429, {})]
    err, text = call("web_search", query="h")
    check("brave: its rate limit counts as temporary",
          err and "unavailable right now" in text and "tell the user" not in text)
    os.remove(keyfile)
    os.environ["SEARCH_BACKENDS"] = "brave"
    n = len(brave.sent)
    err, text = call("web_search", query="i")
    check("SEARCH_BACKENDS naming Brave without a key is a configuration problem, nothing sent",
          err and "configuration problem" in text and len(brave.sent) == n)
    os.environ.pop("SEARCH_BACKENDS")

    os.environ["SEARCH_CACHE_TTL"] = "900"
    pacing(until=0)
    ddg.replies[:] = [Reply(200, RESULTS)]
    call("web_search", query="cached query")
    n = len(ddg.sent)
    err, text = call("web_search", query="Cached  Query")
    check("a repeated query is answered from the cache", not err and len(ddg.sent) == n)

for url in ("http://127.0.0.1:8080/", "http://localhost/"):
    err, text = call("web_fetch", url=url)
    check(f"web_fetch refuses a local address ({url})", err and "non-public" in text)
site = Site()
ss.requests.get = site
site.replies[:] = [Reply(302, "", url="http://1.1.1.1/", location="http://127.0.0.1:8080/")]
err, text = call("web_fetch", url="http://1.1.1.1/")
check("web_fetch refuses a redirect to a local address",
      err and "non-public" in text and len(site.sent) == 1)

print(f"\n{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
