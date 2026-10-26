# Tuning

Two flags do almost all of the work. One is famous, the other is not.

## Run the sweep first

```bash
local-claude bench --tune
```

This starts the real server at each candidate split, times a long-prompt request,
stops at the first value that will not allocate, then tries a doubled micro-batch
(below), and writes the winners to `~/.local-claude/ncmoe.<build>` and
`ubatch.<build>`, where the launcher reads them. It measures the actual server — same context, slots and
KV type you run — because all three consume the VRAM the expert layers compete
for, and `llama-bench` would size its KV cache from its own arguments instead.

Re-run it after changing model, quantisation, context size, KV type, or llama.cpp
build.

## `--n-cpu-moe` — the strategy

Keeps the routed experts of the first N layers in system RAM; everything else
goes to the GPU. Lower N means more expert weights in VRAM and a faster token,
until the allocation fails outright.

Reference machine, 8 GB VRAM, 40-layer model:

| `--n-cpu-moe` | prefill tok/s | decode tok/s | VRAM |
|---|---|---|---|
| 40 (all experts in RAM) | 159.0 | 26.8 | 4115 MiB |
| 38 | 189.2 | 27.7 | 5111 MiB |
| 36 | 198.6 | 28.4 | 6039 MiB |
| **34** | 210.5 | **29.5** | 6967 MiB |
| 32 | — | — | did not start |

Each pair of layers costs ~930 MiB. Tuning from 40 to 34 was worth 10% of decode:
the flag's *existence* is the win, its exact value is polish.

## `--ubatch-size` — the one nobody looks at

Same split, varying the micro-batch:

| `--ubatch-size` | prefill tok/s | decode tok/s |
|---|---|---|
| 512 — llama.cpp's default | 231.9 | 30.5 |
| 1024 | 391.8 | 31.6 |
| **2048** | **648.1** | 31.0 |
| 3072 / 4096 | — | compute buffer no longer fits at this split |

**2.8× prefill, decode untouched.** The 512 default is sized for a model that
lives on the GPU. Here, prompt processing copies the RAM-resident experts to the
GPU once per micro-batch (llama.cpp does this for any batch of 32+ tokens; see
[MOE-OFFLOAD.md](MOE-OFFLOAD.md#the-flag-that-mattered-more-than-the-tuning)), so
a larger micro-batch pays that copy fewer times per token.

4096 fits once more expert layers move to RAM, and that trade pays: on the CUDA
build with pinned threads, `--n-cpu-moe 37 --ubatch-size 4096` reads a 9.5k-token
prompt at 1056 tok/s against 854 at 34 / 2048, and decodes at 38.8 against 39.5.
On a 65k-token prompt it is 1005 against 791 tok/s. So `bench --tune` tunes the
pair: after the split sweep it tries a doubled micro-batch at the smallest split
that qualifies, and keeps it if it gains ≥ 10% prefill for ≤ 5% decode. Until you
tune, the micro-batch comes from `install.sh` (by VRAM).

**Starting is not enough: the tuner keeps VRAM in reserve.** The server allocates
GPU memory after start too — the CUDA backend records a graph for each new batch
shape. `--n-cpu-moe 36 --ubatch-size 4096` started, benchmarked fine and passed
the agentic test with 395 MiB free, then crashed on a later request with
`CUDA error: out of memory`. One more layer in RAM left 855 MiB and survived the
same requests. So `bench --tune` rejects any split that leaves less than
`LOCAL_VRAM_MARGIN` (768 MiB) free after its test request. On the reference
machine the old 34 / 2048 left only 447 MiB and is rejected too.

If something else holds VRAM when the server starts, the launcher says
`retrying at 2048` and starts with the smaller micro-batch; an explicit
`LOCAL_UBATCH` is not second-guessed and fails.

## Threads — one per performance core, pinned

Decode multiplies the RAM-resident experts on the CPU with all threads in
lockstep, so the slowest thread sets the pace. The launcher therefore starts one
thread per performance core and pins each to its own core
(`--cpu-mask … --cpu-strict 1`, chosen by `bin/pick-cpus.sh` from `/sys`). Prefill
is unaffected: it runs on the GPU.

Reference machine (i9-13900HX: 8 P-cores with hyperthreading, 16 E-cores), CUDA
build, decode tok/s on a 9.5k-token prompt, with N busy-loop processes competing
for the CPU:

| Threads | idle | 4 busy | 8 busy | 16 busy |
|---|---|---|---|---|
| 16, unpinned (the old default) | 35.4 | 25–34 | 32.6 | 31.0 |
| 8, pinned one per P-core | **39.2** | **37.2** | **36.5** | **35.1** |
| 8, allowed on P-cores but free to move | 39.9 | 29.3 | 14.4 | 31–35 |

Strict pinning is both faster and steadier. Letting the 8 threads move among the
P-cores is not a compromise: under load they pile onto the same cores. The
busy-loop load is CPU-only; a memory-bandwidth-heavy neighbour is untested.

## Everything else

| Env | Default | When to touch it |
|---|---|---|
| `LOCAL_CTX` | from VRAM (131072 on 8 GB) | Must exceed 100k, or Claude Code overruns the server before it auto-compacts |
| `LOCAL_NCMOE` | tuned value | Override the sweep |
| `LOCAL_UBATCH` | tuned (4096 here), else from VRAM | See above — the single biggest prefill lever |
| `LOCAL_KV_TYPE` | `q8_0` | `f16` doubles KV memory for little quality gain |
| `LOCAL_LOAD_MODE` | unset (memory-mapped) | `none` puts the RAM-resident experts in pinned memory: prefill 1155 vs 1070 tok/s here, decode unchanged. But ~17 GB is then locked while the server runs, so other programs run out of memory instead of the model slowing down. Use it on a machine that does nothing else |
| `LOCAL_SLOTS` | 2 | Slots share one KV pool; the second stops background calls evicting your conversation's prefix |
| `LOCAL_THREADS` | one per performance core | Setting it also turns pinning off — see Threads above |
| `LOCAL_PIN` | 1 | `0` leaves threads unpinned |
| `LOCAL_THINKING` | 1 | `0` disables chain-of-thought — much faster turns, worse reasoning |
| `LOCAL_MODEL_FILE` | best in `~/models` | Pick a model explicitly |
| `LOCAL_MMPROJ` | auto-detected | Vision projector; see MODELS.md |
| `LOCAL_MMPROJ_DEVICE` | `cpu` | `gpu` offloads the vision tower, costing ~1 GB of VRAM the experts want |
| `LOCAL_DEVICE` | autodetected | Force a GPU; never the integrated one |
| `LOCAL_TEMPLATE` | auto-patched | `0` uses the model's stock chat template — see DESIGN-NOTES |
| `LOCAL_SEARCH` | 1 | `0` drops the web-search MCP server, leaving no web access at all: Claude Code's own web tools are off by default |
| `LOCAL_STORAGE` | `separate` | Where local sessions keep transcripts (`/resume`), prompt history, auto-memory and file snapshots (`/rewind`). `separate`: `~/.local-claude/claude`, never seen by cloud sessions. `none`: nothing is kept. `shared`: `~/.claude`, like any Claude Code session, so cloud sessions can see it. Per session |
| `LOCAL_BUILTIN_WEBFETCH` | 0 | `1` allows Claude Code's own WebFetch. It fetches on this machine and summarises with the local model; the launcher skips its domain check, which would otherwise ask Anthropic about every domain. Its WebSearch stays off: Anthropic's servers run those searches, so locally it returns nothing and the model answers from memory |
| `LOCAL_CLAUDE_MD` | `~/.local-claude/CLAUDE.md` | Your standing instructions for local sessions, in every project: the local counterpart of `~/.claude/CLAUDE.md`. Loaded only if the file exists; keep it short, it is sent with every session |
| `LOCAL_GLOBAL_CLAUDE_MD` | 0 | `1` also loads your global `~/.claude/CLAUDE.md`, which local sessions skip by default (~8k tokens on the reference machine) |
| `LOCAL_RULES` | built-in | `""` disables the injected operating rules |
| `LOCAL_BACKEND` | `llamacpp` | `ollama` if you have one running |

### Rarely touched

| Env | Default | What it does |
|---|---|---|
| `LOCAL_HOST` / `LOCAL_PORT` | `127.0.0.1` / `8080` | Where the server listens. Change the port if something else owns 8080 |
| `LOCAL_BATCH` | the larger of 2048 and the micro-batch | Logical batch size. llama.cpp caps `--ubatch-size` to it, so it must not be smaller |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | 32768 | Room for one reply. Too low truncates a long `Write` into invalid JSON — see [TROUBLESHOOTING](TROUBLESHOOTING.md) |
| `API_TIMEOUT_MS` | 1800000 | 30 minutes. Must exceed the worst-case generation: 32k tokens at ~30 tok/s is ~18 minutes |
| `LOCAL_START_TIMEOUT` | 600 | Seconds to wait for the server to come up; a 22 GB model reads from disk on first load |
| `LOCAL_STOP_TIMEOUT` | 60 | Half-seconds to wait for it to exit before `SIGKILL`. It must release its VRAM before the next one starts |
| `LOCAL_NCMOE_FALLBACK` | model's layer count | Expert split used before `bench --tune` has measured anything |
| `LOCAL_VRAM_MARGIN` | 768 | MiB of VRAM `bench --tune` requires free after its test request; raise it if the server still dies mid-session |
| `LOCAL_SEARCH_CONFIG` | `~/.local-claude/search-mcp.json` | Path to the MCP config `install.sh` generates |
| `LOCAL_LLAMA_ARGS` | — | Extra flags passed straight to `llama-server` |
| `LOCAL_CLAUDE_CONFIG_DIR` | `~/.local-claude/claude` | The separate folder `LOCAL_STORAGE=separate` and `none` use |
| `LOCAL_CLAUDE_PREFIX` / `LOCAL_CLAUDE_STATE` / `MODELS_DIR` | see [INSTALL](INSTALL.md#what-goes-where) | Relocate the install, state or models |

## Sizing context

Context competes with expert layers for VRAM, but on a hybrid model it is cheap.
Qwen3.5/3.6-35B-A3B has real attention on only every fourth layer, with 2 KV
heads — about **11 KB per token** at `q8_0`, so a 131k window costs ~1.4 GB. A
conventional model of the same size wants roughly 16× that.

Do not shrink the window to buy expert layers without checking: Claude Code will
not auto-compact below 100k tokens, so a smaller pool means the server rejects a
long session rather than the client shrinking it.

## Backend: CUDA vs Vulkan

`install.sh` picks Vulkan for NVIDIA because llama.cpp ships no Linux CUDA binary
and Vulkan needs nothing installed. A source-built CUDA backend is faster, though.
Measured on the reference machine, same model, same split, same micro-batch, same
9537-token prompt:

| | Vulkan | CUDA |
|---|---|---|
| prefill | 590.7 tok/s | **690.3** |
| decode | 28.5 tok/s | **35.0** |
| VRAM at `--n-cpu-moe 34` | 7420 MiB | 7744 MiB |
| agentic end-to-end | 60 s | **40 s** |

On longer prompts the gap widens: 1143 tok/s prefill on a 14k-token prompt against
761 for Vulkan. CUDA also needs slightly more VRAM for the same split, so re-tune
after switching — and note that the tuned value and the device name are both
cached **per build**, because the same card is `Vulkan1` under one backend and
`CUDA0` under the other.

```bash
./bin/build-cuda.sh --install-nvcc   # add --install-nvcc if you have no compiler
LLAMA_DIR=~/.local/share/local-claude/<build>-cuda local-claude bench --tune
ln -sfn ~/.local/share/local-claude/<build>-cuda ~/.local/share/local-claude/current
```

Getting `nvcc` is the only hard part, and "CUDA is installed" usually does not
mean you have it — see [INSTALL.md](INSTALL.md#platforms).
