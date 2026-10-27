#!/usr/bin/env python3
"""MCP stdio server — web search + fetch through the real `lynx` text browser.

Sites see an ordinary Lynx client: Lynx's own User-Agent, no JS, no CSS,
no images, no headless Chrome fingerprint. The model sees Lynx's ``-dump``
output: plain text with numbered [n] links and a "References" list. The
protocol, the engine chain and the messages are shared with search_server.py
through search_common.py.

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
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import search_common as common
from search_common import (MAX_RESULTS, TEXT_TYPES, DuckDuckGoPacing, SearchSetupError, SearchUnavailable,
                           _brave_key, _brave_key_file, _check_public)

common.LOG_TAG = "[search-lynx]"  # its log lines say which server wrote them

LYNX = os.environ.get("LYNX_BIN") or shutil.which("lynx") or "lynx"
LYNX_TIMEOUT = int(os.environ.get("LYNX_TIMEOUT", "25"))
LYNX_WIDTH = int(os.environ.get("LYNX_WIDTH", "100"))


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


class DuckDuckGoLynx(DuckDuckGoPacing):
    URL = "https://lite.duckduckgo.com/lite/?q={q}"
    # Lynx renders text, so the bot check is recognised by its wording; its
    # HTML (anomaly-modal, challenge-form) never reaches the dump.
    BLOCK_MARKERS = ("bots use duckduckgo too",
                     "confirm this search was made by a human")

    def _is_blocked(self, text: str, results) -> bool:
        if results:
            return False  # a results page, even for a query about "anomalies"
        low = text.lower()
        return any(m in low for m in self.BLOCK_MARKERS)

    def _record(self, text: str, results):
        """Record a bot check or an answer (DuckDuckGoPacing). Lynx shows no HTTP
        status, so an answer is a page with results."""
        self._record_outcome(self._is_blocked(text, results), bool(results))

    def search(self, query: str, k: int = MAX_RESULTS):
        self._turn()
        url = self.URL.format(q=urllib.parse.quote_plus(query))
        text = _tidy(_run_lynx(url))
        results = _parse_ddg_dump(text, k)
        self._record(text, results)
        return results


# ---- keyed / self-hosted backends -------------------------------------------


def _strip_tags(s: str) -> str:
    import re as _re
    return _re.sub(r"<[^>]+>", "", s or "").strip()


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
                                   f"(HTTP {code}). This is a configuration problem: "
                                   "tell the user.")
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
                                   "search.formats in its settings.yml. This is a "
                                   "configuration problem: tell the user.")
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


# ---- web_fetch ----------------------------------------------------------------


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


# ---- the MCP side, shared with the other search server (search_common.py) ----

SERVER_INFO = {"name": "local-search-lynx", "version": "1.0.0"}


def run_search(query: str):
    return common.run_search(query, BACKENDS, (ValueError, RuntimeError))  # lynx failed, bad JSON


def dispatch_tool(name: str, args: dict) -> str:
    return common.dispatch_tool(name, args, run_search, web_fetch)


def handle(req: dict):
    return common.handle(req, dispatch_tool, SERVER_INFO)


def main():
    common.main(handle, BACKENDS)


if __name__ == "__main__":
    main()
