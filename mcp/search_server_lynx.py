#!/usr/bin/env python3
"""
lynx_search_mcp.py - an MCP server that searches and reads the web through the
real `lynx` text browser.

Sites see an ordinary Lynx client: Lynx's own User-Agent, no JavaScript, no CSS,
no images, no headless Chrome fingerprint. Your model sees Lynx's `-dump`
output: plain text with numbered [n] links and a "References" list at the end.

Requirements
    lynx on PATH          (apt install lynx / brew install lynx / choco install lynx)
    pip install mcp

Run
    python lynx_search_mcp.py           # stdio (LM Studio, Claude Desktop, Jan, etc.)
    python lynx_search_mcp.py --http    # streamable HTTP on http://127.0.0.1:8000/mcp

Environment variables (all optional)
    LYNX_BIN          path to the lynx binary
    LYNX_SEARCH_URL   search URL template, {q} is replaced by the encoded query.
                      Default: https://lite.duckduckgo.com/lite/?q={q}
                      Examples:
                        https://www.mojeek.com/search?q={q}
                        http://localhost:8080/search?q={q}      (self-hosted SearXNG)
    LYNX_USER_AGENT   override the User-Agent (unset = Lynx's real one)
    LYNX_TIMEOUT      seconds per request, default 25
    LYNX_WIDTH        dump width in columns, default 100
"""

import os
import re
import shutil
import subprocess
import sys
import urllib.parse

try:                                   # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:                    # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as Server

LYNX = os.environ.get("LYNX_BIN") or shutil.which("lynx") or "lynx"
SEARCH_URL = os.environ.get("LYNX_SEARCH_URL", "https://lite.duckduckgo.com/lite/?q={q}")
USER_AGENT = os.environ.get("LYNX_USER_AGENT")
TIMEOUT = int(os.environ.get("LYNX_TIMEOUT", "25"))
WIDTH = int(os.environ.get("LYNX_WIDTH", "100"))

mcp = Server("lynx-search")


class LynxError(RuntimeError):
    pass


def _check_url(url: str) -> str:
    """Only plain web URLs: blocks file://, lynxexec: and anything that could
    be read by lynx as a command-line option."""
    url = url.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise LynxError(f"Only http(s) URLs are allowed, got: {url!r}")
    return url


def _run_lynx(url: str, include_links: bool = True) -> str:
    cmd = [
        LYNX,
        "-dump",                      # render the page to text and exit
        f"-width={WIDTH}",
        "-display_charset=utf-8",
        "-assume_charset=utf-8",
        "-accept_all_cookies",        # never stop to ask about cookies
        "-hiddenlinks=ignore",        # drop invisible links from the list
        "-unique_urls",               # one reference number per distinct URL
        f"-connect_timeout={TIMEOUT}",
    ]
    if not include_links:
        cmd.append("-nolist")         # text only, no References section
    if USER_AGENT:
        cmd.append(f"-useragent={USER_AGENT}")
    cmd.append(_check_url(url))

    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,  # any interactive prompt gets "no"
            capture_output=True,
            timeout=TIMEOUT + 10,
        )
    except FileNotFoundError:
        raise LynxError(f"lynx not found (looked for {LYNX!r}). Install it or set LYNX_BIN.")
    except subprocess.TimeoutExpired:
        raise LynxError(f"lynx timed out after {TIMEOUT + 10}s fetching {url}")

    out = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0 and not out.strip():
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise LynxError(f"lynx failed (exit {proc.returncode}): {err or 'no output'}")
    return out


_DDG_REDIRECT = re.compile(r"https?://(?:lite\.|html\.)?duckduckgo\.com/l/\?\S+")


def _unwrap_redirects(text: str) -> str:
    """DuckDuckGo wraps result links in /l/?uddg=<real url>; show the real URL."""
    def repl(m: re.Match) -> str:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(m.group(0)).query)
        return qs.get("uddg", [m.group(0)])[0]
    return _DDG_REDIRECT.sub(repl, text)


def _tidy(text: str) -> str:
    text = _unwrap_redirects(text)
    text = re.sub(r"[ \t]+\n", "\n", text)     # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # runs of blank lines
    return text.strip()


def _window(text: str, start: int, max_chars: int) -> str:
    total = len(text)
    start = max(0, min(start, total))
    end = min(total, start + max_chars)
    chunk = text[start:end]
    if start > 0 or end < total:
        note = f"\n\n[showing characters {start}-{end} of {total}"
        note += f"; call again with start={end} for more]" if end < total else "]"
        chunk += note
    return chunk


@mcp.tool()
def web_search(query: str, max_chars: int = 6000) -> str:
    """Search the web with the lynx text browser and return the results page as
    plain text. Result links are numbered [n] and their URLs are listed under
    "References" at the end. Use fetch_page on a URL to read a result."""
    url = SEARCH_URL.replace("{q}", urllib.parse.quote_plus(query))
    try:
        text = _tidy(_run_lynx(url))
    except LynxError as e:
        return f"Search failed: {e}"
    if "bots use duckduckgo too" in text.lower():
        return ("The search engine answered with a bot challenge instead of results. "
                "Wait a while, slow down, or point LYNX_SEARCH_URL at another engine "
                "(e.g. a self-hosted SearXNG).")
    return _window(text, 0, max_chars)


@mcp.tool()
def fetch_page(url: str, start: int = 0, max_chars: int = 10000,
               include_links: bool = True) -> str:
    """Open a web page in the lynx text browser and return it as plain text.
    Long pages come back in windows: pass the `start` value from the footer to
    read the next part. Set include_links=false to drop the References list."""
    try:
        text = _tidy(_run_lynx(url, include_links=include_links))
    except LynxError as e:
        return f"Fetch failed: {e}"
    return _window(text, start, max_chars) or "(page rendered no text)"


if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
