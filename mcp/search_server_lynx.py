#!/usr/bin/env python3
"""MCP stdio server — web search + fetch through the real `lynx` text browser.

Sites see an ordinary Lynx client: Lynx's own User-Agent, no JS, no CSS,
no images, no headless Chrome fingerprint. The model sees Lynx's ``-dump``
output: plain text with numbered [n] links and a "References" list.

Search backends (default: DuckDuckGo lite; configurable via env):

  ddg    DuckDuckGo lite via lynx      default — lynx's real UA rarely triggers
                                        bot checks; DDG pacing is shared via
                                        ddg-pacing.json when present
  searxng  your own SearXNG instance    needs SEARXNG_URL (json format enabled)
  brave    Brave Search API             needs an API key in
                                        ~/.local-claude/brave-search.key

Other knobs (all optional):
  LYNX_BIN           path to the lynx binary (default: finds on PATH)
  LYNX_TIMEOUT       seconds per lynx request, default 25
  LYNX_WIDTH         dump width in columns, default 100
  DDG_MIN_INTERVAL   seconds between DuckDuckGo requests (default 5), counted
                     across every session on this machine
  DDG_JITTER         random extra delay, 0 to this many seconds (default 3)
  DDG_COOLDOWN       first back-off after a bot check, seconds (default 300)
  DDG_MAX_COOLDOWN   max back-off, seconds (default 1800)
  SEARCH_CACHE_TTL   seconds to reuse results (default 900; 0 = off)
  (No WEB_FETCH_MAX_BYTES here: lynx reads a page whole. max_chars, the tool's
  own argument, still limits what reaches the model.)
  WEB_FETCH_ALLOW_PRIVATE=1  let web_fetch reach localhost / private networks
  SEARCH_MCP_LOG     log path, as in search_server.py

Protocol channel is stdout — all logging goes to stderr.
"""
import fcntl
import ipaddress
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from contextlib import contextmanager
from urllib.parse import parse_qs, unquote, urljoin, urlparse

LYNX = os.environ.get("LYNX_BIN") or shutil.which("lynx") or "lynx"
LYNX_TIMEOUT = int(os.environ.get("LYNX_TIMEOUT", "25"))
LYNX_WIDTH = int(os.environ.get("LYNX_WIDTH", "100"))

MAX_RESULTS = 10

def log(*a):
    print("[search-lynx]", *a, file=sys.stderr, flush=True)

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

# ---- Lynx helpers -----------------------------------------------------------

def _run_lynx(url: str, include_links: bool = True, *, follow: bool = True,
              headers: bool = False) -> str:
    """Fetch a URL with lynx -dump and return the rendered text.

    follow=False leaves a redirect unfollowed; headers=True returns the
    response's status line and headers, then its source, instead of the text.
    """
    cmd = [
        LYNX,
        "-mime_header" if headers else "-dump",
        f"-width={LYNX_WIDTH}",
        "-display_charset=utf-8",
        "-assume_charset=utf-8",
        "-accept_all_cookies",
        "-hiddenlinks=ignore",
        "-unique_urls",
        f"-connect_timeout={LYNX_TIMEOUT}",
    ]
    if not include_links:
        cmd.append("-nolist")
    if not follow:
        cmd.append("-noredir")
    cmd.append(url)

    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=LYNX_TIMEOUT + 10,
        )
    except FileNotFoundError:
        raise RuntimeError(f"lynx not found (looked for {LYNX!r}). Install it or set LYNX_BIN.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"lynx timed out after {LYNX_TIMEOUT + 10}s fetching {url}")

    out = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0 and not out.strip():
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"lynx failed (exit {proc.returncode}): {err or 'no output'}")
    return out


def _unwrap_ddg_redirects(text: str) -> str:
    """DuckDuckGo wraps result links in /l/?uddg=<real url>; show the real URL."""
    pattern = re.compile(
        r"https?://(?:lite\.|html\.)?duckduckgo\.com/l/\?[^\s]*"
    )
    def repl(m):
        qs = parse_qs(urlparse(m.group(0)).query)
        return qs.get("uddg", [m.group(0)])[0]
    return pattern.sub(repl, text)


def _tidy(text: str) -> str:
    """Strip trailing whitespace on lines, collapse blank lines."""
    text = _unwrap_ddg_redirects(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_ddg_dump(text: str, k: int = MAX_RESULTS) -> list:
    """Parse DuckDuckGo lite results from lynx -dump output.

    Lynx renders the page as:
        1.  [2]Title here
        snippet text...
            www.python.org/path/
        2.  [3]Another title
        snippet text...
            example.com/path/
        ...
        References
        2. www.python.org/path/
        3. example.com/path/
        ...
    Note: result numbers and link numbers diverge (header/nav links steal numbers).
    """
    lines = text.split("\n")
    results = []
    cur = None

    # Collect result lines: "N.  [M]Title" where N and M may differ
    for line in lines:
        m = re.match(r"^\s*(\d+)\.\s+\[(\d+)\](.+)$", line)
        if m:
            if cur is not None:
                results.append(cur)
            cur = {"title": "", "url": "", "snippet": "", "ref_num": int(m.group(2))}
            cur["title"] = m.group(3).strip()
            continue
        if cur is not None:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip bare URL lines — lynx dumps them inline and in References;
            # URLs are already captured in the ref_urls mapping.
            if re.match(r"^(?:\d+\.\s+)?(?:https?://)?[\w.-]+\.\w+", stripped):
                continue
            if stripped == "References":
                results.append(cur)
                cur = None
                continue
            cur["snippet"] += " " + re.sub(r"[ \t]+$", "", stripped)

    if cur is not None:
        results.append(cur)

    # Collect URLs from the References section
    ref_urls = {}
    in_refs = False
    for line in lines:
        if line.strip() == "References":
            in_refs = True
            continue
        if in_refs:
            stripped = line.strip()
            if not stripped:
                continue  # skip blank lines between References and URLs
            m = re.match(r"^\s*(\d+)\.\s+(https?://\S+)", line)
            if m:
                ref_urls[int(m.group(1))] = m.group(2)
            elif not re.match(r"^\s*\d+\.", stripped):
                in_refs = False

    for r in results:
        r["url"] = ref_urls.get(r["ref_num"], "")
        r["title"] = r["title"].strip()
        r["snippet"] = r["snippet"].strip()

    # Real results are unwrapped from duckduckgo.com/l/?uddg=... by _tidy; ads
    # click through duckduckgo.com/y.js, so a result still on DuckDuckGo is one.
    return [r for r in results
            if r["url"] and not (urlparse(r["url"]).hostname or "").endswith("duckduckgo.com")][:k]


# ---- DuckDuckGo via lynx -----------------------------------------------------

class DuckDuckGoLynx:
    URL = "https://lite.duckduckgo.com/lite/?q={q}"
    # Lynx renders text, so the bot check is recognised by its wording; its
    # HTML (anomaly-modal, challenge-form) never reaches the dump.
    BLOCK_MARKERS = ("bots use duckduckgo too",
                     "confirm this search was made by a human")

    def __init__(self):
        self.min_interval = _env_float("DDG_MIN_INTERVAL", 5.0)
        self.jitter = max(0.0, _env_float("DDG_JITTER", 3.0))
        self.base_cooldown = _env_float("DDG_COOLDOWN", 300.0)
        self.max_cooldown = _env_float("DDG_MAX_COOLDOWN", 1800.0)
        self.own_state = {}

    @contextmanager
    def _state(self):
        """Shared pacing state (file-level lock)."""
        try:
            fh = open(os.path.join(_state_dir(), "ddg-pacing.json"), "a+")
        except OSError:
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
        """Wait for the next allowed request."""
        with self._state() as st:
            now = time.time()
            left = st.get("until", 0) - now
            if 0 < left <= self.max_cooldown:
                raise SearchBlocked(
                    "DuckDuckGo is refusing automated searches from this machine "
                    f"for now. No search is sent to it for {max(1, round(left / 60))} min, "
                    "because searching during the block prolongs it."
                )
            gap = self.min_interval + random.uniform(0.0, self.jitter)
            time.sleep(min(gap, max(0.0, st.get("last", 0) + gap - now)))
            st["last"] = time.time()

    def _is_blocked(self, text: str, results) -> bool:
        if results:
            return False  # a results page, even for a query about "anomalies"
        low = text.lower()
        return any(m in low for m in self.BLOCK_MARKERS)

    def _record(self, text: str, results):
        """Record success or block."""
        blocked = self._is_blocked(text, results)
        with self._state() as st:
            if blocked:
                st["strikes"] = min(st.get("strikes", 0) + 1, 16)
                cool = min(self.base_cooldown * 2 ** (st["strikes"] - 1), self.max_cooldown)
                st["until"] = time.time() + cool
            elif results:
                st["strikes"] = 0
        # Raised only once the state is written back: raised inside the `with`,
        # it skipped the write, so the pause was never recorded.
        if blocked:
            log(f"duckduckgo served its bot check; pausing for {cool:.0f}s "
                f"(block #{st['strikes']} in a row)")
            raise SearchBlocked(
                "DuckDuckGo is refusing automated searches from this machine "
                f"for now. No search is sent to it for {max(1, round(cool / 60))} min."
            )

    def search(self, query: str, k: int = MAX_RESULTS):
        self._turn()
        url = self.URL.format(q=urllib.parse.quote_plus(query))
        text = _tidy(_run_lynx(url))
        results = _parse_ddg_dump(text, k)
        self._record(text, results)
        return results


# ---- Shared backend infrastructure (same as search_server.py) ----------------

class SearchUnavailable(Exception):
    """A backend could not answer for now: blocked, rate-limited or down."""

class SearchBlocked(SearchUnavailable):
    """DuckDuckGo answered with its bot check instead of results."""

class SearchSetupError(Exception):
    """A backend is set up wrongly: the user has to fix something."""

def _strip_tags(s: str) -> str:
    import re as _re
    return _re.sub(r"<[^>]+>", "", s or "").strip()

def _brave_key_file() -> str:
    return os.environ.get("SEARCH_BRAVE_KEY_FILE") or os.path.join(_state_dir(), "brave-search.key")

def _brave_key() -> str:
    try:
        with open(_brave_key_file()) as fh:
            return fh.read().strip()
    except OSError:
        return ""

def brave_search(query: str, k: int = MAX_RESULTS):
    key = _brave_key()
    url = ("https://api.search.brave.com/res/v1/web/search"
           f"?q={urllib.parse.quote_plus(query)}&count={min(k, 20)}")
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "Lynx/2.8.5 dev",
        "X-Subscription-Token": key,
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        code = e.code
        if code == 429:
            raise SearchUnavailable(
                "Brave Search refused the search (HTTP 429): its rate limit, "
                "or the account's monthly credit is used up.")
        try:
            err_body = json.loads(e.read().decode())
        except Exception:
            err_body = {}
        if code in (401, 403) or err_body.get("error", {}).get("code") == "SUBSCRIPTION_TOKEN_INVALID":
            raise SearchSetupError(f"Brave Search rejected the API key in {_brave_key_file()} "
                                   f"(HTTP {code}). This is a configuration problem.")
        raise SearchUnavailable(f"Brave Search answered HTTP {code}.")
    except Exception as e:
        raise SearchUnavailable(f"Brave Search failed: {e}")

    items = (body.get("web") or {}).get("results") or []
    return [{"title": _strip_tags(i.get("title", "")), "url": i["url"],
             "snippet": _strip_tags(i.get("description", ""))}
            for i in items if i.get("url")][:k]

def searxng_search(query: str, k: int = MAX_RESULTS):
    base = os.environ["SEARXNG_URL"].rstrip("/")
    url = f"{base}/search?q={urllib.parse.quote_plus(query)}&format=json"
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise SearchSetupError("SearXNG refused format=json; add `json` under "
                                   "search.formats in its settings.yml.")
        raise SearchUnavailable(f"SearXNG answered HTTP {e.code}.")
    except Exception as e:
        raise SearchUnavailable(f"SearXNG failed: {e}")

    items = data.get("results") or []
    if not items and data.get("unresponsive_engines"):
        names = ", ".join(str(e[0]) if isinstance(e, (list, tuple)) and e else str(e)
                          for e in data["unresponsive_engines"])
        raise SearchUnavailable(f"SearXNG's engines are failing ({names}).")
    return [{"title": _strip_tags(i.get("title", "")), "url": i["url"],
             "snippet": _strip_tags(i.get("content", ""))}
            for i in items if i.get("url")][:k]

BACKENDS = {
    "ddg":     ("DuckDuckGo", lambda: None, DuckDuckGoLynx().search),
    "searxng": ("SearXNG", lambda: None if os.environ.get("SEARXNG_URL") else "SEARXNG_URL",
                searxng_search),
    "brave":   ("Brave Search", lambda: None if _brave_key() else f"an API key in {_brave_key_file()}",
                brave_search),
}

def _order():
    chosen = os.environ.get("SEARCH_BACKENDS", "").strip()
    return [b.strip().lower() for b in (chosen or "ddg,searxng,brave").split(",") if b.strip()], bool(chosen)

# ---- cache -------------------------------------------------------------------

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

def run_search(query: str):
    """Return (backend_name, results)."""
    key = _cache_key(query)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    order, chosen = _order()
    errors, empty_from = [], None
    for name in order:
        try:
            if name not in BACKENDS:
                raise SearchSetupError(f"SEARCH_BACKENDS names '{name}', which is not a search "
                                       "backend (use ddg, searxng or brave).")
            label, lacks, fn = BACKENDS[name]
            missing = lacks()
            if missing and not chosen:
                continue
            if missing:
                raise SearchSetupError(f"SEARCH_BACKENDS names {label}, which needs {missing}. "
                                       "This is a configuration problem.")
            res = fn(query, MAX_RESULTS)
        except SearchSetupError as e:
            log(f"backend {name} set up wrongly")
            raise SearchSetupError(" ".join(errors + [f"Web search could not run: {e}"]))
        except SearchUnavailable as e:
            log(f"backend {name} unavailable: {e}")
            errors.append(str(e))
            continue
        except (ValueError, RuntimeError) as e:  # lynx errors, etc.
            log(f"backend {name} failed: {type(e).__name__}")
            errors.append(f"{label} could not be reached ({type(e).__name__}).")
            continue
        if res:
            _cache_put(key, (name, res))
            return name, res
        empty_from = empty_from or name

    if empty_from:
        _cache_put(key, (empty_from, []))
        return empty_from, []

    raise SearchUnavailable("Web search is unavailable right now; this is not an empty "
                            "result. " + " ".join(errors) + " Try again later, or "
                            "web_fetch a URL you already know.")

# ---- web_fetch ----------------------------------------------------------------

TEXT_TYPES = ("text/", "html", "xml", "json", "javascript")

def _check_public(url: str):
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
                             f"address ({ip}). Set WEB_FETCH_ALLOW_PRIVATE=1.")

def _probe(url: str):
    """One request, redirects unfollowed: (HTTP status or 0, lower-cased headers)."""
    head = re.split(r"\r?\n\r?\n", _run_lynx(url, follow=False, headers=True), maxsplit=1)[0]
    lines = head.splitlines()
    m = re.match(r"HTTP/\S+\s+(\d{3})", lines[0] if lines else "")
    hdrs = {}
    for line in lines[1:]:
        k, sep, v = line.partition(":")
        if sep:
            hdrs[k.strip().lower()] = v.strip()
    return (int(m.group(1)) if m else 0), hdrs


def web_fetch(url: str, max_chars: int = 6000) -> str:
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    # lynx would follow a redirect on its own, past the public-address check, so
    # each hop is requested unfollowed and its target checked before the next.
    try:
        for _ in range(6):
            _check_public(url)
            status, hdrs = _probe(url)
            if status in (301, 302, 303, 307, 308) and hdrs.get("location"):
                url = urljoin(url, hdrs["location"])
                continue
            break
        else:
            raise ValueError("too many redirects")
        ctype = hdrs.get("content-type", "").lower()
        if ctype and not any(t in ctype for t in TEXT_TYPES):
            size = hdrs.get("content-length")
            return (f"(not a text page: {ctype.split(';')[0].strip()}"
                    f"{f', {size} bytes' if size else ''}; nothing extracted)")
        raw = _run_lynx(url, follow=False)
    except RuntimeError as e:
        raise ValueError(f"fetch failed: {e}")
    text = _tidy(raw)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + (f"\n\n[...truncated at {max_chars} chars; refine "
                                   "with a more specific fetch if needed]")
    return text or "(page rendered no extractable text)"

# ---- dispatch -----------------------------------------------------------------

def dispatch_tool(name: str, args: dict) -> str:
    logpath = os.environ.get("SEARCH_MCP_LOG")
    if logpath:
        try:
            with open(logpath, "a") as fh:
                detail = f" {json.dumps(args)[:200]}" if os.environ.get("LOCAL_SEARCH_LOG_QUERIES") == "1" else ""
                fh.write(f"{time.strftime('%H:%M:%S')} {name}{detail}\n")
        except OSError:
            pass
    if name == "web_search":
        query = str(args["query"]).strip()
        if not query:
            raise ValueError("query is empty")
        k = max(1, min(int(args.get("max_results", 5) or 5), MAX_RESULTS))
        source, res = run_search(query)
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
        return web_fetch(str(args["url"]), max_chars)
    raise ValueError(f"unknown tool: {name}")

# ---- MCP tool declarations --------------------------------------------------

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

# ---- JSON-RPC dispatch ------------------------------------------------------

def handle(req: dict):
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        pv = (req.get("params") or {}).get("protocolVersion", "2024-11-05")
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": pv,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "local-search-lynx", "version": "1.0.0"}}}
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
            text = dispatch_tool(name, args)
            is_error = False
        except Exception as e:
            log(f"tool {name} failed: {type(e).__name__}")
            text = f"Error running {name}: {e}"
            is_error = True
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "content": [{"type": "text", "text": text}], "isError": is_error}}
    if rid is not None:
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return None

def main():
    usable = [b for b in _order()[0] if b in BACKENDS and not BACKENDS[b][1]()]
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
        resp = handle(req)
        if resp is not None and req.get("id") is not None:
            send(resp)
    log("stdin closed, exiting")

if __name__ == "__main__":
    main()
