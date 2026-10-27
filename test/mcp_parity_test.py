#!/usr/bin/env python3
"""The two search servers side by side: what they share must behave the same.

mcp/search_server.py (requests) and mcp/search_server_lynx.py differ only in
how they reach DuckDuckGo and fetch pages. The MCP protocol, the engine chain,
the pacing after a bot check and every message the model sees are meant to be
identical, so each scenario here runs on both and compares. Offline: every
search engine is a stand-in.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

STATE = tempfile.mkdtemp()
KEY = os.path.join(STATE, "brave-search.key")
os.environ.update(LOCAL_CLAUDE_STATE=STATE, SEARCH_BRAVE_KEY_FILE=KEY, SEARCH_CACHE_TTL="0",
                  DDG_MIN_INTERVAL="0", DDG_JITTER="0")
for var in ("SEARCH_BACKENDS", "SEARXNG_URL", "SEARCH_MCP_LOG"):
    os.environ.pop(var, None)
MCP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp")
sys.path.insert(0, MCP)
try:
    import search_server as R
except ImportError as e:
    print(f"SKIP: the requests server needs requests and bs4 ({e})")
    sys.exit(0)
import search_server_lynx as L  # noqa: E402

SERVERS = {"requests": R, "lynx": L}
fails = 0


def check(name, cond, detail=""):
    global fails
    print(("PASS: " if cond else "FAIL: ") + name)
    if not cond and detail:
        print("      " + detail.replace("\n", "\n      "))
    fails += not cond


def same(name, got, expect=None):
    """Both servers gave the same answer (and the expected one, if given)."""
    a, b = got["requests"], got["lynx"]
    ok = a == b and (expect is None or expect(a))
    check(name, ok, "" if ok else f"requests: {a!r}\nlynx:     {b!r}")


# ---- the MCP exchange, each server run as the config runs it --------------------
# From another folder, as Claude Code does: the server must find what it imports.
EXCHANGE = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {"jsonrpc": "2.0", "id": 3, "method": "ping"},
    {"jsonrpc": "2.0", "id": 4, "method": "no/such/method"},
]
got = {}
for name, mod in SERVERS.items():
    p = subprocess.run([sys.executable, mod.__file__], cwd=STATE, capture_output=True, text=True, timeout=60,
                       input="".join(json.dumps(m) + "\n" for m in EXCHANGE))
    replies = [json.loads(line) for line in p.stdout.splitlines() if line.strip()]
    for r in replies:
        r.get("result", {}).pop("serverInfo", None)  # each server names itself
    got[name] = replies
same("protocol: the same replies to initialize, tools/list, ping and an unknown method", got,
     lambda r: [m.get("id") for m in r] == [1, 2, 3, 4]
     and r[0]["result"]["protocolVersion"] == "2025-06-18" and r[3]["error"]["code"] == -32601)

# ---- the engine chain, engines replaced by stand-ins ----------------------------
asked, behave = [], {}


def stand_in(engine, mod):
    def search(query, k):
        asked.append(engine)
        if behave.get(engine) == "refuse":
            raise mod.SearchUnavailable(f"{engine} refused for now.")
        if behave.get(engine) == "badkey":
            raise mod.SearchSetupError(f"{engine} rejected the key.")
        return [{"title": f"{engine} {i}", "url": f"https://{engine}.example/{i}", "snippet": f"about {i}"}
                for i in range(3)]
    return search


for mod in SERVERS.values():
    for engine in ("ddg", "searxng", "brave"):
        label, lacks, _ = mod.BACKENDS[engine]
        mod.BACKENDS[engine] = (label, lacks, stand_in(engine, mod))


def search(mod, backends=None, key=False, searxng=False, **how):
    """Run one search; what came back, which engines were asked, and the message."""
    asked.clear(); behave.clear(); behave.update(how)
    if backends is None:
        os.environ.pop("SEARCH_BACKENDS", None)
    else:
        os.environ["SEARCH_BACKENDS"] = backends
    if key:
        open(KEY, "w").write("k")
    elif os.path.exists(KEY):
        os.remove(KEY)
    if searxng:
        os.environ["SEARXNG_URL"] = "http://searx.example"
    else:
        os.environ.pop("SEARXNG_URL", None)
    r = mod.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                    "params": {"name": "web_search", "arguments": {"query": "q", "max_results": 2}}})["result"]
    return r["isError"], list(asked), r["content"][0]["text"]


CHAIN = [  # (scenario, arguments, expected engines asked, an error?)
    ("nothing set up: DuckDuckGo", {}, ["ddg"], False),
    ("DuckDuckGo refuses: Brave answers", dict(key=True, ddg="refuse"), ["ddg", "brave"], False),
    ("all set up, DuckDuckGo refuses: SearXNG next", dict(key=True, searxng=True, ddg="refuse"), ["ddg", "searxng"], False),
    ("SEARCH_BACKENDS=brave,ddg: Brave first", dict(backends="brave,ddg", key=True), ["brave"], False),
    ("SEARCH_BACKENDS names Brave without a key", dict(backends="brave"), [], True),
    ("SEARCH_BACKENDS names SearXNG without a URL", dict(backends="searxng"), [], True),
    ("SEARCH_BACKENDS names an unknown engine", dict(backends="google"), [], True),
    ("a rejected key stops the search", dict(backends="ddg,brave", key=True, ddg="refuse", brave="badkey"), ["ddg", "brave"], True),
    ("every engine refuses", dict(key=True, ddg="refuse", brave="refuse"), ["ddg", "brave"], True),
]
for scenario, kw, engines, error in CHAIN:
    same(f"chain: {scenario}", {n: search(m, **kw) for n, m in SERVERS.items()},
         lambda r, engines=engines, error=error: r[0] == error and r[1] == engines)
os.environ.pop("SEARCH_BACKENDS", None)


def call(mod, tool, **args):
    r = mod.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                    "params": {"name": tool, "arguments": args}})["result"]
    return r["isError"], r["content"][0]["text"]


LOG = os.path.join(STATE, "search-mcp.log")


def logged(mod, **env):
    """Whether a search, in this environment, leaves a line in the search log."""
    behave.clear()
    if os.path.exists(LOG):
        os.remove(LOG)
    saved = {k: os.environ.get(k) for k in ("SEARCH_MCP_LOG", "LOCAL_STORAGE", "LOCAL_SEARCH_LOG_QUERIES")}
    os.environ.update(SEARCH_MCP_LOG=LOG, **env)
    try:
        call(mod, "web_search", query="q")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return os.path.exists(LOG) and os.path.getsize(LOG) > 0


same("log: the search log records which tool ran", {n: logged(m) for n, m in SERVERS.items()},
     lambda r: r)
same("log: under LOCAL_STORAGE=none there is no search log",
     {n: logged(m, LOCAL_STORAGE="none") for n, m in SERVERS.items()}, lambda r: not r)
same("log: ...unless LOCAL_SEARCH_LOG_QUERIES=1 asks for one",
     {n: logged(m, LOCAL_STORAGE="none", LOCAL_SEARCH_LOG_QUERIES="1") for n, m in SERVERS.items()},
     lambda r: r)
same("output: an empty query is an error", {n: call(m, "web_search", query="  ") for n, m in SERVERS.items()},
     lambda r: r[0])
same("output: an unknown tool is an error", {n: call(m, "no_such_tool") for n, m in SERVERS.items()},
     lambda r: r[0])
for url in ("http://127.0.0.1:8080/", "http://localhost/"):
    same(f"web_fetch: refuses a local address ({url})", {n: call(m, "web_fetch", url=url) for n, m in SERVERS.items()},
         lambda r: r[0] and "non-public" in r[1])

# ---- DuckDuckGo's bot check, through each server's own transport ----------------
PACING = os.path.join(STATE, "ddg-pacing.json")


class Reply:
    def __init__(self, status, text):
        self.status_code, self.text, self.url = status, text, R.DuckDuckGo.URL

    def raise_for_status(self):
        pass


requests_pages = {"bot": Reply(202, '<form id="challenge-form">'),
                  "results": Reply(200, '<a class="result-link" href="https://example.com/a">A</a>'
                                        '<td class="result-snippet">about a</td>')}
lynx_pages = {"bot": "Unfortunately, bots use DuckDuckGo too.\n",
              "results": "   1.  [1]A\n       about a\n\nReferences\n\n   1. https://example.com/a\n"}
sent = []


def serve(page):
    """Make both servers' DuckDuckGo requests answer with this page."""
    def post(url, data=None, **kw):
        sent.append("requests")
        return requests_pages[page]

    def run_lynx(url, include_links=True, **kw):
        sent.append("lynx")
        return lynx_pages[page]
    R.DDG.session.post = post
    L._run_lynx = run_lynx


def pacing():
    try:
        with open(PACING) as fh:
            return json.loads(fh.read() or "{}")
    except OSError:
        return {}


def bot_check(mod):
    """A bot check, then a search during the pause: the two messages, the state, requests sent."""
    os.environ["SEARCH_BACKENDS"] = "ddg"
    open(PACING, "w").close()
    for engine in ("ddg",):
        label, lacks, _ = mod.BACKENDS[engine]
        mod.BACKENDS[engine] = (label, lacks, mod.DDG.search if mod is R else mod.DuckDuckGoLynx().search)
    sent.clear()
    serve("bot")
    first = call(mod, "web_search", query="q1")
    st = pacing()
    during = call(mod, "web_search", query="q2")
    n = len(sent)
    with open(PACING, "w") as fh:
        json.dump(dict(st, until=0), fh)
    serve("results")
    after = call(mod, "web_search", query="q3")
    return (first, during, n, st.get("strikes"), 290 < st.get("until", 0) - time.time() <= 300,
            after[0], pacing().get("strikes"))


same("pacing: a bot check pauses DuckDuckGo, says so in the same words, and results end it",
     {n: bot_check(m) for n, m in SERVERS.items()},
     lambda r: r[0][0] and r[1][0] and r[2] == 1 and r[3] == 1 and r[4] and not r[5] and r[6] == 0)

subprocess.run(["rm", "-rf", STATE])
print(f"\n{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
