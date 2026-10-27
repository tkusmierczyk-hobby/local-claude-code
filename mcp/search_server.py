#!/usr/bin/env python3
"""Minimal MCP stdio server giving the local agent web search + fetch.

No third-party MCP SDK: speaks JSON-RPC 2.0 over newline-delimited stdio
(the MCP stdio transport) directly. Only deps are requests + bs4. The
protocol, the engine chain and the messages are shared with
search_server_lynx.py through search_common.py.

Search backends are tried in order (SEARCH_BACKENDS, default
"ddg,searxng,brave"). One that isn't set up is skipped, unless SEARCH_BACKENDS
names it; one that fails for now (bot check, rate limit, down) falls through
to the next; one that is set up wrongly (a rejected key) stops the search and
is reported, rather than the query silently going elsewhere:

  ddg      DuckDuckGo lite, keyless     best effort: requests are paced,
                                        results cached, and after a bot check
                                        it stops contacting DuckDuckGo for a
                                        cooldown instead of retrying into it
  searxng  your own SearXNG instance    needs SEARXNG_URL (with the json
                                        format enabled in settings.yml)
  brave    Brave Search API             needs an API key in
                                        ~/.local-claude/brave-search.key
                                        (or SEARCH_BRAVE_KEY_FILE)

Other knobs (all optional):
  DDG_MIN_INTERVAL   seconds between DuckDuckGo requests (default 5), counted
                     across every session on this machine
  DDG_JITTER         random extra delay on top of that, 0 to this many seconds
                     (default 3), so gaps are 5-8 s rather than a fixed beat
  DDG_COOLDOWN       first back-off after a bot check, seconds (default 300);
                     doubles on each further block, up to DDG_MAX_COOLDOWN
                     (default 1800), and resets after a successful search
  SEARCH_CACHE_TTL   seconds to reuse results for a repeated query (default
                     900; 0 turns caching off)
  WEB_FETCH_MAX_BYTES  download cap for web_fetch (default 2000000)
  WEB_FETCH_ALLOW_PRIVATE=1  let web_fetch reach localhost / private networks
  SEARCH_MCP_LOG, LOCAL_SEARCH_LOG_QUERIES  call log, as before

Protocol channel is stdout — all logging goes to stderr.
"""
import fcntl
import json
import os
import random
import re
import time
from contextlib import contextmanager
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import search_common as common
from search_common import (MAX_RESULTS, TEXT_TYPES, SearchBlocked, SearchSetupError, SearchUnavailable,
                           _brave_key, _brave_key_file, _check_public, _ddg_refusing, _env_float,
                           _state_dir, log)

UAS = [
    # The standard UA first: it is the default, kept instead of rotating
    # every request.
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Lynx/2.8.5 dev liberational HTML browser",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36",
]

_cached_ua: str | None = None


def _ua() -> str:
    """Return the current User-Agent; stick with it until a block forces rotation."""
    global _cached_ua
    if _cached_ua is not None:
        return _cached_ua
    # Pick the first available UA; on block we advance to the next one.
    _cached_ua = UAS[0]
    return _cached_ua


def _rotate_ua() -> str:
    """Try the next UA in the list; cache whichever we land on."""
    global _cached_ua
    if _cached_ua is None:
        return _ua()
    try:
        idx = UAS.index(_cached_ua) + 1
    except ValueError:
        idx = 0
    if idx >= len(UAS):
        return _cached_ua
    _cached_ua = UAS[idx]
    return _cached_ua


def _strip_tags(s: str) -> str:
    # Brave/SearXNG snippets can carry <strong> highlighting.
    return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)


# ---- DuckDuckGo (keyless, best effort) --------------------------------------

def _decode_ddg_href(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    if "uddg=" in href:
        q = parse_qs(urlparse(href).query)
        if "uddg" in q:
            return unquote(q["uddg"][0])
    return href


def parse_ddg_lite(html: str, k: int = MAX_RESULTS):
    """Pair each result link with the snippet that follows it in the page.

    Walking links and snippets in document order (rather than zipping two
    separate lists by index) keeps snippets aligned when a result has no
    snippet, and lets sponsored rows be dropped without shifting the rest.
    """
    soup = BeautifulSoup(html, "html.parser")
    out, cur = [], None
    for el in soup.select("a.result-link, .result-snippet"):
        if "result-link" in (el.get("class") or []):
            cur = None
            if el.find_parent(class_="result-sponsored"):
                continue
            url = _decode_ddg_href(el.get("href", ""))
            host = urlparse(url).hostname or ""
            # Ads click through duckduckgo.com/y.js rather than a uddg= wrapper.
            if not url.startswith(("http://", "https://")) or host.endswith("duckduckgo.com"):
                continue
            cur = {"title": el.get_text(" ", strip=True), "url": url, "snippet": ""}
            out.append(cur)
        elif cur is not None and not cur["snippet"]:
            cur["snippet"] = el.get_text(" ", strip=True)
    return out[:k]


class DuckDuckGo:
    URL = "https://lite.duckduckgo.com/lite/"
    BLOCK_MARKERS = ("anomaly-modal", "anomaly.js", 'id="challenge-form"',
                     "bots use duckduckgo too")

    def __init__(self):
        self.session = requests.Session()  # keeps cookies and the connection
        self.session.headers.update({"User-Agent": _ua(),
                                     "Accept-Language": "en-US,en;q=0.9"})
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

    def _record(self, r):
        """A bot check starts a pause, longer each time it is still there; a result ends that."""
        blocked = self._is_blocked(r)
        if blocked:
            # The session's header was set once, at start: it has to follow the
            # rotation itself, or DuckDuckGo keeps seeing the blocked one.
            old = self.session.headers["User-Agent"]
            self.session.headers["User-Agent"] = _rotate_ua()
            log(f"UA {old!r} blocked by DuckDuckGo; next request uses "
                f"{self.session.headers['User-Agent']!r}")
        with self._state() as st:
            if blocked:
                st["strikes"] = min(st.get("strikes", 0) + 1, 16)
                cool = min(self.base_cooldown * 2 ** (st["strikes"] - 1), self.max_cooldown)
                st["until"] = time.time() + cool
            elif r.status_code == 200:
                st["strikes"] = 0
        if blocked:
            log(f"duckduckgo served its bot check; pausing it for {cool:.0f}s "
                f"(block #{st['strikes']} in a row)")
            raise SearchBlocked(_ddg_refusing(cool))

    def _is_blocked(self, r) -> bool:
        if "result-link" in r.text:
            return False  # a results page, even for a query about "anomalies"
        if r.status_code in (202, 403, 429):
            return True
        host = urlparse(r.url).hostname or ""
        if not host.endswith("duckduckgo.com"):
            return True  # redirected away to a challenge host
        low = r.text.lower()
        return any(m in low for m in self.BLOCK_MARKERS)

    def search(self, query: str, k: int = MAX_RESULTS):
        self._turn()
        # POST, as the lite page's own search form does. One request per search:
        # each one counts against the limit.
        r = self.session.post(self.URL, data={"q": query}, timeout=20)
        self._record(r)
        r.raise_for_status()
        return parse_ddg_lite(r.text, k)


DDG = DuckDuckGo()


# ---- keyed / self-hosted backends -------------------------------------------


def brave_search(query: str, k: int = MAX_RESULTS):
    r = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": min(k, 20)},
        headers={"Accept": "application/json", "User-Agent": _ua(),
                 "X-Subscription-Token": _brave_key()},
        timeout=20)
    if r.status_code == 429:
        raise SearchUnavailable("Brave Search refused the search (HTTP 429): its rate limit, "
                                "or the account's monthly credit is used up.")
    try:
        body = r.json() if r.status_code >= 400 else {}
    except ValueError:
        body = {}
    err = (body.get("error") or {}) if isinstance(body, dict) else {}
    if r.status_code in (401, 403) or err.get("code") == "SUBSCRIPTION_TOKEN_INVALID":
        raise SearchSetupError(f"Brave Search rejected the API key in {_brave_key_file()} "
                               f"(HTTP {r.status_code}). This is a configuration problem: "
                               "tell the user.")
    if r.status_code >= 400:
        raise SearchUnavailable(f"Brave Search answered HTTP {r.status_code}.")
    items = (r.json().get("web") or {}).get("results") or []
    return [{"title": _strip_tags(i.get("title", "")), "url": i["url"],
             "snippet": _strip_tags(i.get("description", ""))}
            for i in items if i.get("url")][:k]


def searxng_search(query: str, k: int = MAX_RESULTS):
    base = os.environ["SEARXNG_URL"].rstrip("/")
    r = requests.get(f"{base}/search", params={"q": query, "format": "json"},
                     headers={"Accept": "application/json", "User-Agent": _ua()},
                     timeout=25)
    if r.status_code == 403:
        raise SearchSetupError("SearXNG refused format=json; add `json` under "
                               "search.formats in its settings.yml. This is a "
                               "configuration problem: tell the user.")
    if r.status_code >= 400:
        raise SearchUnavailable(f"SearXNG answered HTTP {r.status_code}.")
    data = r.json()
    items = data.get("results") or []
    if not items and data.get("unresponsive_engines"):
        # Its upstream engines (often including DuckDuckGo) are blocking it too.
        names = ", ".join(str(e[0]) if isinstance(e, (list, tuple)) and e else str(e)
                          for e in data["unresponsive_engines"])
        raise SearchUnavailable(f"SearXNG's engines are failing ({names}).")
    return [{"title": _strip_tags(i.get("title", "")), "url": i["url"],
             "snippet": _strip_tags(i.get("content", ""))}
            for i in items if i.get("url")][:k]


# name: (label, what it lacks to run or None, search function)
BACKENDS = {
    "ddg":     ("DuckDuckGo", lambda: None, DDG.search),
    "searxng": ("SearXNG", lambda: None if os.environ.get("SEARXNG_URL") else "SEARXNG_URL",
                searxng_search),
    "brave":   ("Brave Search", lambda: None if _brave_key() else f"an API key in {_brave_key_file()}",
                brave_search),
}


# ---- web_fetch ----------------------------------------------------------------


def _charset(ctype: str):
    m = re.search(r"charset=[\"']?([\w.:-]+)", ctype, re.I)
    return m.group(1) if m else None


def web_fetch(url: str, max_chars: int = 6000) -> str:
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    cap = int(_env_float("WEB_FETCH_MAX_BYTES", 2_000_000))

    # Follow redirects by hand so every hop gets the public-address check.
    for _ in range(6):
        _check_public(url)
        r = requests.get(url, headers={"User-Agent": _ua()}, timeout=25,
                         stream=True, allow_redirects=False)
        if r.is_redirect:
            nxt = urljoin(url, r.headers.get("location", ""))
            r.close()
            url = nxt
            continue
        break
    else:
        raise ValueError("too many redirects")

    with r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()
        if ctype and not any(t in ctype for t in TEXT_TYPES):
            size = r.headers.get("content-length")
            return (f"(not a text page: {ctype.split(';')[0].strip()}"
                    f"{f', {size} bytes' if size else ''}; nothing extracted)")
        chunks, total = [], 0
        for chunk in r.iter_content(65536):
            chunks.append(chunk)
            total += len(chunk)
            if total >= cap:
                break
        raw = b"".join(chunks)[:cap]
        truncated_bytes = total >= cap

    if not ctype and b"\x00" in raw[:2048]:
        return "(not a text page: binary content; nothing extracted)"

    if "html" in ctype or b"<html" in raw[:2000].lower():
        # Bytes in, so bs4 can honour the page's own <meta charset> when the
        # header doesn't name one (requests would otherwise guess ISO-8859-1).
        soup = BeautifulSoup(raw, "html.parser", from_encoding=_charset(ctype))
        for tag in soup(["script", "style", "noscript", "header", "footer",
                         "nav", "aside", "form", "svg", "iframe"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = main.get_text("\n", strip=True)
    else:
        text = raw.decode(_charset(ctype) or "utf-8", errors="replace")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + (f"\n\n[...truncated at {max_chars} chars; refine "
                                   "with a more specific fetch if needed]")
    elif truncated_bytes:
        text += f"\n\n[...page was larger than {cap} bytes; only the start was read]"
    return text or "(page had no extractable text)"


# ---- the MCP side, shared with the other search server (search_common.py) ----

SERVER_INFO = {"name": "local-search", "version": "1.1.0"}


def run_search(query: str):
    return common.run_search(query, BACKENDS, (requests.RequestException, ValueError))  # network, bad JSON


def dispatch_tool(name: str, args: dict) -> str:
    return common.dispatch_tool(name, args, run_search, web_fetch)


def handle(req: dict):
    return common.handle(req, dispatch_tool, SERVER_INFO)


def main():
    common.main(handle, BACKENDS)


if __name__ == "__main__":
    main()
