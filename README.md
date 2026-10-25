# local-claude

Run Claude Code against a model on your own machine. Same CLI, same tools,
permissions, hooks and subagents — but inference happens locally, and nothing
contacts Anthropic at runtime. The only outside traffic is web search when the
model uses it: queries go to DuckDuckGo and pages are fetched from the sites
themselves (`LOCAL_SEARCH=0` turns that off).

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

## Tests

```bash
./test/preflight.sh          # install sanity — no model needed
./test/api_compat_test.sh    # /v1/messages conformance against a running server
./test/e2e_agentic_test.sh   # the real agentic loop in a sandbox
./test/mcp_server_test.py    # MCP handshake for the web-search server
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
| Web search and fetch through a local server; Claude Code's own web tools are off, since its fetch asks Anthropic about every domain and its search cannot work locally | `LOCAL_SEARCH=0` for no web access at all; `LOCAL_BUILTIN_WEBFETCH=1` allows the built-in fetch, with the Anthropic check skipped |
| Your global `~/.claude/CLAUDE.md` is skipped: it is written for the cloud model and costs context | `LOCAL_GLOBAL_CLAUDE_MD=1` to load it anyway |
| Instructions of your own for every local session, written for the local model | `~/.local-claude/CLAUDE.md`, if you create it |
| None of the conversation is stored: Claude Code's transcripts and prompt history are off for local sessions, and the search log records only which tool ran | `LOCAL_SAVE_SESSIONS=1` keeps sessions, which `/resume` needs |
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
- The default backend is Vulkan, because llama.cpp ships no Linux CUDA binary and
  Vulkan needs nothing installed. A source-built CUDA backend is ~20% faster on
  decode and ~50% on prefill — `./bin/build-cuda.sh --install-nvcc`, see
  [docs/TUNING.md](docs/TUNING.md#backend-cuda-vs-vulkan).

## License

MIT — see [LICENSE](LICENSE).

This is an independent project, not affiliated with or endorsed by Anthropic.
Claude Code is not included; install it separately, under Anthropic's terms.
