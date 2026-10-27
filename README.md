# local-claude

Run Claude Code against a model on your own machine. Same CLI, same tools,
permissions, hooks and subagents — but inference happens locally, and nothing
contacts Anthropic at runtime. The only outside traffic is web search when the
model uses it: queries go to the search engines you enable (DuckDuckGo, needing
no setup, by default) and pages are fetched from the sites themselves — see
[Web search](#web-search); `LOCAL_SEARCH=0` turns it off.

The interesting part is what makes it usable: a 35-billion-parameter model on an
8 GB graphics card, at a speed you can actually work with. That is not
compression, it is placement — see [how it works](#how-it-works).

```
claude  (Claude Code CLI)
  │  ANTHROPIC_BASE_URL=http://127.0.0.1:8080
  ▼
llama-server  (llama.cpp — user-space, no root)
  ├─ GPU   attention · KV cache · router · shared experts
  └─ RAM   routed expert weights            --n-cpu-moe N
```

## Requirements

- **Claude Code** already installed and on your `PATH`
- **Linux x64** (tested). The installer also has a macOS/arm64 path, but it has
  never been run on a Mac
- **A GPU helps but is not required.** NVIDIA via Vulkan needs no CUDA toolkit
- **RAM is the real constraint** — it decides which model you can run:

| Usable RAM | What you can run | Download |
|---|---|---|
| 30 GB+ | Qwen3.6-35B-A3B, Q4_K_XL | 22.4 GB |
| 24 GB | same model, Q3_K_XL | 16.9 GB |
| 21 GB | Gemma-4-26B-A4B | 14.3 GB |
| less | small models only — fine for testing the plumbing | 1.3 GB |

`./install.sh` works this out for you and prints the right command.

## Quickstart

```bash
git clone <this-repo> local-claude && cd local-claude
./install.sh                      # detects hardware, installs llama.cpp, verifies

# fetch a model (slow and resumable — run it detached)
setsid nohup local-claude pull unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450 \
  </dev/null >/dev/null 2>&1 &
local-claude models               # watch progress

local-claude bench --tune         # measure the best expert split for your GPU
local-claude                      # start coding
```

Installs go to `~/.local/share/local-claude`, `~/.local-claude` and `~/models`.
Nothing needs root, and [uninstalling](docs/INSTALL.md#uninstalling) is three
`rm -rf`s.

## Commands

```bash
local-claude                      # interactive REPL on the local model
local-claude -p "prompt"          # print mode
local-claude status               # what is loaded, how it is split, VRAM/RAM
local-claude models               # local models + download progress
local-claude start / stop         # manage the server without launching Claude
local-claude bench --tune         # sweep --n-cpu-moe and --ubatch-size, store both
local-claude pull <repo> <file> [sha256]
local-claude init                 # scaffold hooks + CLAUDE.md into another project
local-claude attempts N "task"    # best-of-N, gated on .claude/verify.sh
```

## How it works

A Mixture-of-Experts model stores far more than it reads. Qwen3.6-35B-A3B holds
35B parameters but routes each token through only 8 of 256 experts per layer, so
about 3B parameters are read per token. Generation speed is set by memory
bandwidth, so that ratio is what makes a 35B model viable on consumer hardware at
all.

That leaves the question of which weights get the fast memory. The usual answer —
"offload some layers to the CPU" — fills VRAM with expert weights that any given
token has a ~3% chance of reading, while evicting attention and the KV cache,
which are read *in full, every token*. Splitting by **tensor role** instead, via
llama.cpp's `--n-cpu-moe`, keeps attention and KV on the GPU and streams the
experts from RAM. Same VRAM, roughly 2.9× more useful bytes read from it per
token.

The full argument, with the arithmetic and where it turned out to be wrong, is in
**[docs/MOE-OFFLOAD.md](docs/MOE-OFFLOAD.md)**.

## Measured

Reference machine: RTX 4060 Laptop (8 GB), i9-13900HX, 31 GB RAM,
Qwen3.6-35B-A3B UD-Q4_K_XL at `--n-cpu-moe 34 --ubatch-size 2048 --ctx-size 131072`.

| | CUDA build | Vulkan build |
|---|---|---|
| Agentic task end to end | **40 s** | 60 s |
| Decode | **34.6 tok/s** | 28.7 tok/s |
| Prefill, 14k-token prompt | **1143 tok/s** | 761 tok/s |
| Context | 131,072 | 131,072 |
| VRAM | 7744 MiB of 8188 | 7420 MiB |
| API conformance | 7/7 | 7/7 |

Both run the same model at the same split; the difference is the backend. CUDA is
worth ~20% of decode and ~50% of prefill here, but it is a source build — see
[docs/INSTALL.md](docs/INSTALL.md#platforms). Vulkan needs nothing installed and
is what `install.sh` sets up by default.

Re-tuned since, with one thread pinned per performance core and
`--n-cpu-moe 37 --ubatch-size 4096`, the CUDA build decodes at ~38 tok/s, reads a
65k-token prompt at ~1000 tok/s, and does the agentic task in about 30 s — see
[docs/TUNING.md](docs/TUNING.md).

**Two flags did most of the work, and the famous one was not the bigger.**
`--n-cpu-moe` is the strategy — without it, none of this runs at a usable speed.
But `--ubatch-size` was worth **2.8× prefill** (232 → 648 tok/s) at no cost to
decode: llama.cpp defaults it to 512, which assumes a GPU-resident model, but here
each micro-batch of prompt copies the RAM-resident experts to the GPU once, so a
larger one copies far less per token. `bench --tune` picks it together with the
split.

## Vision

The recommended model is multimodal. Fetch the 0.9 GB projector and the launcher
picks it up automatically — screenshots and image files then work in Claude Code:

```bash
local-claude pull unsloth/Qwen3.6-35B-A3B-GGUF mmproj-F16.gguf \
  8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f
```

Details and the VRAM trade-off: [docs/MODELS.md](docs/MODELS.md#vision).

## Web search

The model gets two tools, `web_search` and `web_fetch`, from a small search
server that runs on your machine. Claude Code's own web tools are switched off:
its WebFetch asks Anthropic about every domain before fetching, and its
WebSearch runs on Anthropic's servers, so against a local model it returns
nothing.

**What leaves your machine:** each search goes to the engine that answers it,
and each fetched page comes from its site. Nothing goes to Anthropic. The
server's log records which tool ran, not the query or the URL
(`LOCAL_SEARCH_LOG_QUERIES=1` logs those too), and with `LOCAL_STORAGE=none` there
is no such log.

### Search engines

Engines are tried in order, `SEARCH_BACKENDS` (default `ddg,searxng,brave`):

- one you have not set up is skipped;
- one that refuses for now (a bot check, a rate limit, an outage) passes the
  query to the next;
- one that is set up wrongly (a rejected key) stops the search with a message,
  rather than the query silently going elsewhere.

| Engine | Setup | Cost | Notes |
|---|---|---|---|
| DuckDuckGo | none | free | No API: the server reads its "lite" web page, best effort. Refuses bursts of searches — see [known issues](#known-issues) |
| SearXNG | your own instance: set `SEARXNG_URL` and add `json` under `search.formats` in its `settings.yml` | free | A metasearch engine that queries Google, Bing and others for you. Its engines can be blocked too |
| Brave Search API | an API key in `~/.local-claude/brave-search.key`, `chmod 600` | a card, and a monthly credit worth about 1,000 searches (October 2026) | Its own index, through an official API. A file rather than an environment variable, which every program you start, cloud sessions included, could read |

`local-claude status` shows the order the engines will be tried in.
`SEARCH_BACKENDS=ddg local-claude` leaves the others out for one session;
`LOCAL_SEARCH=0` removes web access altogether.

**Why not Google or Bing directly?** Neither offers a usable search API any
more. Google's Custom Search JSON API is
[closed to new customers and ends on 1 January 2027](https://developers.google.com/custom-search/v1/overview),
and its whole-web search is for partners only. Microsoft
[retired the Bing Search APIs in August 2025](https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement);
their replacement works only inside Azure's AI agents. Scraping either breaks
their terms and runs into CAPTCHAs. For their results, use SearXNG.

### Two search servers

Both offer the same tools, engines, order and messages
(`test/mcp_parity_test.py` checks this). They differ in how they reach
DuckDuckGo and fetch pages:

| | `requests` (default) | `lynx` |
|---|---|---|
| Needs | Python's `requests` and `beautifulsoup4` | the `lynx` text browser |
| DuckDuckGo and fetched pages | Python, with a browser's User-Agent that changes after a bot check | `lynx`, so sites see an ordinary Lynx client |
| Page size | `web_fetch` stops at `WEB_FETCH_MAX_BYTES` (2 MB) | reads pages whole; two requests per page, the first to check where it redirects |

`LOCAL_SEARCH_BACKEND=lynx local-claude` uses the lynx server for one session;
set it when running `./install.sh` to make it the default. Brave and SearXNG are
plain API calls in both.

### Known issues

- **DuckDuckGo's bot check.** Since autumn 2026 DuckDuckGo answers a burst of
  searches (others measured about nine in a row) with a bot check that covers
  your whole IP address, and searching while it is up keeps it there. So the
  server sends DuckDuckGo at most one search every 5–8 seconds and, after a bot
  check, none for 5 minutes, doubling while the check persists, up to 30. All
  sessions on the machine share this. Meanwhile searches go to SearXNG or Brave
  if you set one up; otherwise web search is unavailable until the pause ends.
  Alternatively, using lynx with **LOCAL_SEARCH_BACKEND=lynx local-claude** does not cause any problems with DuckDuckGo.
  Don't test it by searching again and again.
- **The model may retell that as a configuration problem.** The tool says the
  bot check is *not* one; a configuration problem (a rejected Brave key, a
  SearXNG without `json`) says so explicitly and asks the model to tell you.
- **`web_fetch` reads text only.** There is no JavaScript, so pages built in the
  browser come back nearly empty, and PDFs and other binary files are reported,
  not read. Local and private addresses are refused, at every redirect
  (`WEB_FETCH_ALLOW_PRIVATE=1` allows them).
- **Repeated queries are answered from a cache** for 15 minutes in each session
  (`SEARCH_CACHE_TTL`).

Every search option is in [docs/TUNING.md](docs/TUNING.md); fixes are in
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#web-search-does-not-work).

## Tests

```bash
./test/preflight.sh          # install sanity — no model needed
./test/api_compat_test.sh    # /v1/messages conformance against a running server
./test/e2e_agentic_test.sh   # the real agentic loop in a sandbox
./test/mcp_server_test.py    # MCP handshake for the web-search server
./test/mcp_parity_test.py    # both web-search servers behave the same
./test/storage_test.sh       # what a session leaves on disk, per LOCAL_STORAGE
./test/bench.sh --tune       # throughput + expert-split sweep
```

## What you also get

Small local models plan worse and claim success more readily than a frontier
model. The stack compensates automatically rather than hoping:

| | |
|---|---|
| Operating rules injected into the system prompt | `LOCAL_RULES=""` to disable |
| Syntax errors fed straight back to the model | `hooks/syntax_check.sh` on Write/Edit |
| Cannot declare "done" while `.claude/verify.sh` fails | `hooks/verify_on_stop.sh`, max 2 retries |
| Web search and fetch through a local server, with Claude Code's own web tools off — see [Web search](#web-search) | `LOCAL_SEARCH=0` for no web access at all; `LOCAL_BUILTIN_WEBFETCH=1` allows the built-in fetch, with the Anthropic check skipped; `LOCAL_BUILTIN_WEBSEARCH=1` the built-in search |
| No background traffic to Anthropic, no feedback reports, no mirroring to claude.ai, remote control or messages from other sessions | `LOCAL_BACKGROUND_TRAFFIC=1`, `LOCAL_FEEDBACK=1`, `LOCAL_REMOTE=1`. Any Claude Code variable or `--settings` key you set yourself also wins over these defaults |
| Your global `~/.claude/CLAUDE.md` is skipped: it is written for the cloud model and costs context | `LOCAL_GLOBAL_CLAUDE_MD=1` to load it anyway |
| Instructions of your own for every local session, written for the local model | `~/.local-claude/CLAUDE.md`, if you create it |
| Local sessions keep their history and memory in a folder of their own, `~/.local-claude/claude`, never in `~/.claude`, where cloud sessions would pick them up | `LOCAL_STORAGE=none` keeps no transcript, history or memory, and no search or server log; a few housekeeping files remain, with nothing of the conversation in them ([details](docs/TUNING.md#everything-else)). `LOCAL_STORAGE=shared` uses `~/.claude` |
| Best-of-N attempts gated on your acceptance test | `local-claude attempts 3 "..."` |

## Documentation

| | |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | platforms, options, uninstalling |
| [docs/MODELS.md](docs/MODELS.md) | model matrix, quantisation, switching |
| [docs/TUNING.md](docs/TUNING.md) | every knob and when to touch it |
| [docs/MOE-OFFLOAD.md](docs/MOE-OFFLOAD.md) | why this works — the arithmetic |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | when it doesn't |
| [docs/DESIGN-NOTES.md](docs/DESIGN-NOTES.md) | what was tried and rejected |
| [CLAUDE.md](CLAUDE.md) | notes for agents working on this repo |

## Limitations

- First request after a server start loads the model — expect up to a minute.
- Features that require Anthropic's cloud (claude.ai connectors, account
  features) are inactive. The agentic loop — files, bash, permissions, subagents,
  hooks, search — is fully local.
- Claude Code prints `unrecognized_model` for any local model. Cosmetic.
- Web search is best effort without setup: DuckDuckGo, the one engine that
  needs none, pauses after a burst of searches. See [Web search](#web-search).
- The default backend is Vulkan, because llama.cpp ships no Linux CUDA binary and
  Vulkan needs nothing installed. A source-built CUDA backend is ~20% faster on
  decode and ~50% on prefill — `./bin/build-cuda.sh --install-nvcc`, see
  [docs/TUNING.md](docs/TUNING.md#backend-cuda-vs-vulkan).

## License

Copyright (C) 2026 tkusmierczyk.

Licensed under the GNU Affero General Public License, version 3 — see
[LICENSE](LICENSE). You may use, change and share it; if you distribute a
modified version, or let others use one over a network, you must publish its
source under the same license.

If those terms do not fit your use, for example in a closed-source or commercial
product, a separate commercial license is available: open an issue to ask.

This is an independent project, not affiliated with or endorsed by Anthropic.
Claude Code is not included; install it separately, under Anthropic's terms.
