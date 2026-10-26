#!/usr/bin/env python3
"""Minimal MCP stdio server giving the local agent web search + fetch.

No third-party MCP SDK: speaks JSON-RPC 2.0 over newline-delimited stdio
(the MCP stdio transport) directly. Search uses DuckDuckGo's keyless "lite"
endpoint; fetch strips HTML to readable text. Only deps are requests + bs4,
already present in this environment.

Protocol channel is stdout — all logging goes to stderr.
"""
import json
import os
import re
import sys
import time
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def log(*a):
    print("[search-mcp]", *a, file=sys.stderr, flush=True)


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


# ---- tool implementations ---------------------------------------------------

def _decode_ddg_href(href: str) -> str:
    if "uddg=" in href:
        q = parse_qs(urlparse("https:" + href if href.startswith("//") else href).query)
        if "uddg" in q:
            return unquote(q["uddg"][0])
    return href


class SearchBlocked(Exception):
    """DuckDuckGo answered with its bot check instead of results."""


def _blocked(r) -> bool:
    # After a burst of searches DuckDuckGo serves a challenge page ("bots use
    # DuckDuckGo too"): HTTP 202, an anomaly-modal form, no result links. A
    # results page is a 200 with result links, even for a query about anomalies.
    # It may also refuse outright with a 403 "error getting results" page.
    return r.status_code == 403 or (
        "result-link" not in r.text and (r.status_code == 202 or "anomaly-modal" in r.text))


# Models paraphrase this to the user, and a vaguer version came out as a claim
# that the user's configuration blocked search, so say what it is and is not.
BLOCKED_MSG = ("DuckDuckGo is temporarily refusing automated searches from this machine "
               "(its rate limit after a burst of searches). This is not a configuration "
               "problem and not an empty result. Try again in a few minutes, or web_fetch "
               "a URL you already know.")


def ddg_search(query: str, max_results: int = 5):
    def parse(html):
        soup = BeautifulSoup(html, "html.parser")
        links = soup.select("a.result-link")
        snippets = soup.select(".result-snippet")
        out = []
        for i, a in enumerate(links[:max_results]):
            out.append({
                "title": a.get_text(" ", strip=True),
                "url": _decode_ddg_href(a.get("href", "")),
                "snippet": snippets[i].get_text(" ", strip=True) if i < len(snippets) else "",
            })
        return out

    hdr = {"User-Agent": UA}
    # GET first (works from probing); POST fallback if the form is returned empty
    r = requests.get("https://lite.duckduckgo.com/lite/", params={"q": query},
                     headers=hdr, timeout=20)
    if _blocked(r):
        raise SearchBlocked(BLOCKED_MSG)
    r.raise_for_status()
    res = parse(r.text)
    if not res:
        r = requests.post("https://lite.duckduckgo.com/lite/", data={"q": query},
                          headers=hdr, timeout=20)
        if _blocked(r):
            raise SearchBlocked(BLOCKED_MSG)
        r.raise_for_status()
        res = parse(r.text)
    return res


def web_fetch(url: str, max_chars: int = 6000) -> str:
    if not re.match(r"^https?://", url):
        url = "https://" + url
    r = requests.get(url, headers={"User-Agent": UA}, timeout=25)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    body = r.text
    if "html" in ctype or "<html" in body[:1000].lower():
        soup = BeautifulSoup(body, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer",
                         "nav", "aside", "form", "svg", "iframe"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = main.get_text("\n", strip=True)
    else:
        text = body
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[...truncated at {max_chars} chars; refine with a more specific fetch if needed]"
    return text or "(page had no extractable text)"


def dispatch_tool(name: str, args: dict) -> str:
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
        query = args["query"]
        k = max(1, min(int(args.get("max_results", 5) or 5), 10))
        res = ddg_search(query, k)
        if not res:
            return f"No results found for: {query}"
        lines = [f"Web search results for: {query}", ""]
        for i, r in enumerate(res, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet']}")
            lines.append("")
        return "\n".join(lines).strip()
    if name == "web_fetch":
        return web_fetch(args["url"], int(args.get("max_chars", 6000) or 6000))
    raise ValueError(f"unknown tool: {name}")


# ---- MCP tool declarations --------------------------------------------------

TOOLS = [
    {
        "name": "web_search",
        "description": ("Search the web (DuckDuckGo) for current information. Returns a "
                        "numbered list of results with title, URL, and snippet. Use for "
                        "up-to-date facts, library docs, error messages, or anything not "
                        "already in the local codebase."),
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
                        "Use after web_search to read a specific result page."),
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
            "serverInfo": {"name": "local-search", "version": "1.0.0"}}}
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


def main():
    log("started")
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
