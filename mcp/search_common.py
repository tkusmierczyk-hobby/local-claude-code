#!/usr/bin/env python3
"""What the two search servers share: the MCP side, the engine chain, the rules.

mcp/search_server.py (requests) and mcp/search_server_lynx.py (the lynx
browser) differ only in how they reach DuckDuckGo and fetch pages. Everything
else lives here, once, so the two cannot drift apart: the JSON-RPC protocol and
tool definitions, the order engines are tried in and how a failure falls
through, the result cache, the public-address check for web_fetch, and the
messages the model sees. Standard library only: the lynx server needs neither
requests nor bs4. test/mcp_parity_test.py checks that both servers agree.
"""
import fcntl
import ipaddress
import json
import os
import random
import socket
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager
from urllib.parse import urlparse


MAX_RESULTS = 10  # backends always fetch this many; the cache serves any k <= 10


LOG_TAG = "[search-mcp]"  # search_server_lynx.py sets its own


def log(*a):
    print(LOG_TAG, *a, file=sys.stderr, flush=True)


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _state_dir() -> str:
    return os.environ.get("LOCAL_CLAUDE_STATE") or os.path.expanduser("~/.local-claude")


# ---- errors --------------------------------------------------------------------

class SearchUnavailable(Exception):
    """A backend could not answer for now: blocked, rate-limited or down.
    Messages never include the query, since they are logged."""


class SearchBlocked(SearchUnavailable):
    """DuckDuckGo answered with its bot check instead of results."""


class SearchSetupError(Exception):
    """A backend is set up wrongly: the user has to fix something."""


# Models retell these to the user, and a vaguer version came out as a claim that
# the user's configuration blocked search, so say what it is and what it is not.


def _ddg_refusing(seconds: float) -> str:
    return ("DuckDuckGo is refusing automated searches from this machine for now (its rate "
            "limit after a burst of searches; not a configuration problem). No search is "
            f"sent to it for {max(1, round(seconds / 60))} min, because searching during "
            "the block prolongs it.")


# ---- DuckDuckGo's pacing, shared by every session on this machine ------------

class DuckDuckGoPacing:
    """When DuckDuckGo may be asked next, and how long it stays paused after a bot check.

    The servers differ in how they ask: each one's DuckDuckGo class calls
    _turn() before a request and _record_outcome() after it.
    """

    def __init__(self):
        self.min_interval = _env_float("DDG_MIN_INTERVAL", 5.0)
        self.jitter = max(0.0, _env_float("DDG_JITTER", 3.0))
        self.base_cooldown = _env_float("DDG_COOLDOWN", 300.0)
        self.max_cooldown = _env_float("DDG_MAX_COOLDOWN", 1800.0)
        self.own_state = {}  # stands in for the shared file when it can't be opened

    @contextmanager
    def _state(self):
        """The pacing state {"last", "until", "strikes"}, held exclusively.

        DuckDuckGo limits the IP address, which every session's server on this
        machine shares, so they share this too, through a file holding only times.
        """
        try:
            fh = open(os.path.join(_state_dir(), "ddg-pacing.json"), "a+")
        except OSError:  # no state folder: pace this process on its own
            yield self.own_state
            return
        with fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            try:
                st = json.loads(fh.read() or "{}")
            except ValueError:
                st = {}
            if not isinstance(st, dict):
                st = {}
            yield st
            fh.truncate(0)
            json.dump(st, fh)

    def _turn(self):
        """Wait for this machine's next turn to send a request; SearchBlocked while paused."""
        with self._state() as st:
            now = time.time()
            left = st.get("until", 0) - now
            if 0 < left <= self.max_cooldown:  # bounded: a clock set back can't stall search
                raise SearchBlocked(_ddg_refusing(left))
            # One gate for every request: bursts are what trigger the bot check.
            # A fresh random gap each turn, so the requests don't land on a fixed beat.
            gap = self.min_interval + random.uniform(0.0, self.jitter)
            time.sleep(min(gap, max(0.0, st.get("last", 0) + gap - now)))
            st["last"] = time.time()

    def _record_outcome(self, blocked: bool, answered: bool):
        """A bot check starts a pause, longer each time it is still there; an answer ends that.

        What counts as either is the server's call: the requests server sees an
        HTTP status, the lynx server only the page's text.
        """
        with self._state() as st:
            if blocked:
                st["strikes"] = min(st.get("strikes", 0) + 1, 16)
                cool = min(self.base_cooldown * 2 ** (st["strikes"] - 1), self.max_cooldown)
                st["until"] = time.time() + cool
            elif answered:
                st["strikes"] = 0
        # Raised only once the state is written back: raised inside the `with`,
        # it skipped the write, so the pause was never recorded.
        if blocked:
            log(f"duckduckgo served its bot check; pausing it for {cool:.0f}s "
                f"(block #{st['strikes']} in a row)")
            raise SearchBlocked(_ddg_refusing(cool))


# ---- Brave's key, engine order, cache and the engine chain ---------------------

def _brave_key_file() -> str:
    return os.environ.get("SEARCH_BRAVE_KEY_FILE") or os.path.join(_state_dir(), "brave-search.key")


def _brave_key() -> str:
    # A file rather than an environment variable, which every program started
    # from the shell, cloud sessions included, could read. Read per search, so
    # a new key works without restarting the session.
    try:
        with open(_brave_key_file()) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _order():
    """The backends to try, in order, and whether the user named them (SEARCH_BACKENDS)."""
    chosen = os.environ.get("SEARCH_BACKENDS", "").strip()
    return [b.strip().lower() for b in (chosen or "ddg,searxng,brave").split(",") if b.strip()], bool(chosen)


_cache: "OrderedDict[str, tuple]" = OrderedDict()


_CACHE_MAX = 256


def _cache_key(query: str) -> str:
    return " ".join(query.split()).lower()


def _cache_get(key):
    ttl = _env_float("SEARCH_CACHE_TTL", 900.0)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < ttl:
        _cache.move_to_end(key)
        return hit[1]
    _cache.pop(key, None)
    return None


def _cache_put(key, value):
    if _env_float("SEARCH_CACHE_TTL", 900.0) <= 0:
        return
    _cache[key] = (time.monotonic(), value)
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def run_search(query: str, backends: dict, transport_errors: tuple):
    """Return (backend_name, results).

    backends maps a name to (label, what it lacks to run or None, search
    function); transport_errors are the exceptions that mean an engine could
    not be reached, which differ with how a server reaches them.

    Raises SearchUnavailable if every backend failed for now, and SearchSetupError
    as soon as one is set up wrongly: the user should hear of a broken key rather
    than have the query silently go elsewhere.
    """
    key = _cache_key(query)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    order, chosen = _order()
    errors, empty_from = [], None
    for name in order:
        try:
            if name not in backends:
                raise SearchSetupError(f"SEARCH_BACKENDS names '{name}', which is not a search "
                                       "backend (use ddg, searxng or brave). This is a "
                                       "configuration problem: tell the user.")
            label, lacks, fn = backends[name]
            missing = lacks()
            if missing and not chosen:
                continue
            if missing:
                raise SearchSetupError(f"SEARCH_BACKENDS names {label}, which needs {missing}. "
                                       "This is a configuration problem: tell the user.")
            res = fn(query, MAX_RESULTS)
        except SearchSetupError as e:
            log(f"backend {name} set up wrongly")
            raise SearchSetupError(" ".join(errors + [f"Web search could not run: {e}"]))
        except SearchUnavailable as e:
            log(f"backend {name} unavailable: {e}")
            errors.append(str(e))
            continue
        except transport_errors as e:  # network, bad JSON
            log(f"backend {name} failed: {type(e).__name__}")  # message can hold the URL
            errors.append(f"{label} could not be reached ({type(e).__name__}).")
            continue
        if res:
            _cache_put(key, (name, res))
            return name, res
        empty_from = empty_from or name  # try the next backend before giving up

    if empty_from:
        _cache_put(key, (empty_from, []))
        return empty_from, []

    raise SearchUnavailable("Web search is unavailable right now; this is not an empty "
                            "result. " + " ".join(errors) + " Try again later, or "
                            "web_fetch a URL you already know.")


# ---- web_fetch's public-address check ------------------------------------------

TEXT_TYPES = ("text/", "html", "xml", "json", "javascript")


def _check_public(url: str):
    """Refuse URLs that resolve to loopback, private, link-local or other
    non-public addresses: the agent follows links from pages it has read, and a
    page can steer it at local services or cloud metadata endpoints.
    (A DNS answer that changes between this check and the request could still
    slip through; this is a guard rail, not a sandbox.)"""
    if os.environ.get("WEB_FETCH_ALLOW_PRIVATE") == "1":
        return
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("only http(s) URLs can be fetched")
    host = p.hostname
    if not host:
        raise ValueError("URL has no host")
    try:
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ValueError(f"could not resolve host {host}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError(f"refusing to fetch {host}: it resolves to a non-public "
                             f"address ({ip}). Set WEB_FETCH_ALLOW_PRIVATE=1 to allow "
                             "local and private URLs.")


# ---- the MCP side: tools, dispatch, JSON-RPC -----------------------------------

def dispatch_tool(name: str, args: dict, search, fetch) -> str:
    """Run a tool: search(query) is the server's engine chain, fetch(url, max_chars) its web_fetch."""
    logpath = os.environ.get("SEARCH_MCP_LOG")
    if logpath:
        try:
            with open(logpath, "a") as fh:
                # The tool name only: queries and URLs come from the user's
                # conversation. LOCAL_SEARCH_LOG_QUERIES=1 logs them as well.
                detail = f" {json.dumps(args)[:200]}" if os.environ.get("LOCAL_SEARCH_LOG_QUERIES") == "1" else ""
                fh.write(f"{time.strftime('%H:%M:%S')} {name}{detail}\n")
        except OSError:
            pass
    if name == "web_search":
        query = str(args["query"]).strip()
        if not query:
            raise ValueError("query is empty")
        k = max(1, min(int(args.get("max_results", 5) or 5), MAX_RESULTS))
        source, res = search(query)
        res = res[:k]
        if not res:
            return f"No results found for: {query}"
        lines = [f"Web search results for: {query} (via {source})", ""]
        for i, r in enumerate(res, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r.get("snippet"):
                lines.append(f"   {r['snippet']}")
            lines.append("")
        return "\n".join(lines).strip()
    if name == "web_fetch":
        max_chars = max(200, min(int(args.get("max_chars", 6000) or 6000), 100_000))
        return fetch(str(args["url"]), max_chars)
    raise ValueError(f"unknown tool: {name}")


TOOLS = [
    {
        "name": "web_search",
        "description": ("Search the web for current information. Returns a numbered list "
                        "of results with title, URL, and snippet. Use for up-to-date facts, "
                        "library docs, error messages, or anything not already in the local "
                        "codebase. Repeated queries are answered from a short cache; if it "
                        "reports search as unavailable, don't retry immediately."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the search query"},
                "max_results": {"type": "integer", "description": "results to return (default 5, max 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": ("Fetch a URL and return its readable text (HTML stripped, truncated). "
                        "Use after web_search to read a specific result page. Public "
                        "http(s) pages only; binary files such as PDFs are not extracted."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "the URL to fetch"},
                "max_chars": {"type": "integer", "description": "max characters to return (default 6000)"},
            },
            "required": ["url"],
        },
    },
]


def handle(req: dict, dispatch, server_info: dict):
    """The reply to one JSON-RPC message, or None for a notification."""
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        pv = (req.get("params") or {}).get("protocolVersion", "2024-11-05")
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": pv,
            "capabilities": {"tools": {}},
            "serverInfo": server_info}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "prompts/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"prompts": []}}
    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"resources": []}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            text = dispatch(name, args)
            is_error = False
        except Exception as e:  # surface as a tool error, not a protocol crash
            log(f"tool {name} failed: {type(e).__name__}")  # the message can hold the URL
            text = f"Error running {name}: {e}"
            is_error = True
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "content": [{"type": "text", "text": text}], "isError": is_error}}
    if rid is not None:  # unknown *request*
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return None  # unknown notification → ignore


def main(respond, backends: dict):
    """Serve MCP over stdio until stdin closes; respond is the server's handle()."""
    usable = [b for b in _order()[0] if b in backends and not backends[b][1]()]
    log(f"started; search backends: {', '.join(usable) or 'none'}")
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            log("bad json line ignored")
            continue
        resp = respond(req)
        if resp is not None and req.get("id") is not None:
            send(resp)
    log("stdin closed, exiting")
