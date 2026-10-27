#!/usr/bin/env python3
"""mcp/search_server_lynx.py offline: DuckDuckGo-shaped pages rendered by the real lynx.

The pages are local files, so nothing here contacts a search engine; lynx itself
renders them, so the parser is checked against real lynx output.
"""
import json
import os
import shutil
import sys
import tempfile

if not shutil.which("lynx"):
    print("SKIP: lynx is not installed")
    sys.exit(0)

STATE = tempfile.mkdtemp()
os.environ.update(LOCAL_CLAUDE_STATE=STATE, SEARCH_BACKENDS="ddg", SEARCH_CACHE_TTL="0",
                  DDG_MIN_INTERVAL="0", DDG_JITTER="0")
os.environ.pop("SEARCH_MCP_LOG", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp"))
import search_server_lynx as L  # noqa: E402

# Shaped like lite.duckduckgo.com: the query echoed in the form, numbered rows,
# a redirect-wrapped link, and a sponsored row.
RESULTS = """<html><body>
<form action="/lite/" method="post"><input type="text" name="q" value="anomaly detection"></form>
<table>
<tr><td>1.&nbsp;</td><td><a href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fwiki%2FAnomaly_detection&amp;rut=abc" class='result-link'>Anomaly detection - Wikipedia</a></td></tr>
<tr><td>&nbsp;</td><td class='result-snippet'>In data analysis, <b>anomaly</b> detection is the identification of rare items.</td></tr>
<tr><td>&nbsp;</td><td><span class='link-text'>en.example.org/wiki/Anomaly_detection</span></td></tr>
<tr class="result-sponsored"><td>2.&nbsp;</td><td><a href="https://duckduckgo.com/y.js?ad_domain=ads.example&amp;u3=x" class='result-link'>Buy anomaly tools</a></td></tr>
<tr class="result-sponsored"><td>&nbsp;</td><td class='result-snippet'>Ad text</td></tr>
<tr><td>3.&nbsp;</td><td><a href="https://example.com/pyod" class='result-link'>PyOD: outlier detection in Python</a></td></tr>
<tr><td>&nbsp;</td><td class='result-snippet'>A library for detecting anomalous objects.</td></tr>
<tr><td>&nbsp;</td><td><span class='link-text'>example.com/pyod</span></td></tr>
</table></body></html>"""

# DuckDuckGo's bot check: only its wording survives lynx, not its HTML.
BOT_CHECK = """<html><body>
<form id="challenge-form" action="//duckduckgo.com/anomaly.js?sv=lite" method="POST">
<div class="anomaly-modal__title">Unfortunately, bots use DuckDuckGo too.</div>
<div class="anomaly-modal__description">Please complete the following challenge to confirm this search was made by a human.</div>
<button type="submit" class="anomaly-modal__submit">Submit</button>
</form></body></html>"""

pages = {}
for name, html in (("results", RESULTS), ("bot_check", BOT_CHECK)):
    pages[name] = os.path.join(STATE, name + ".html")
    with open(pages[name], "w") as fh:
        fh.write(html)

real_run_lynx = L._run_lynx
served = []  # the page each DuckDuckGo request got


def serve(name):
    """Answer DuckDuckGo requests with a local page, rendered by real lynx."""
    def run(url, include_links=True):
        served.append(name)
        return real_run_lynx(pages[name], include_links)
    L._run_lynx = run


def call(tool, **args):
    r = L.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": tool, "arguments": args}})["result"]
    return r["isError"], r["content"][0]["text"]


def pacing():
    try:
        with open(os.path.join(STATE, "ddg-pacing.json")) as fh:
            return json.loads(fh.read() or "{}")
    except OSError:
        return {}


fails = 0


def check(name, cond):
    global fails
    print(("PASS: " if cond else "FAIL: ") + name)
    fails += not cond


serve("results")
err, text = call("web_search", query="anomaly detection")
check("a results page about anomalies is not taken for a bot check",
      not err and "https://example.com/pyod" in text)

serve("bot_check")
err, text = call("web_search", query="python")
check("DuckDuckGo's bot check is recognised from its wording",
      err and "refusing automated searches" in text)

shutil.rmtree(STATE)
print(f"\n{'ALL PASSED' if fails == 0 else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
