# Tuning

Two flags do almost all of the work. One is famous, the other is not.

## Run the sweep first

```bash
local-claude bench --tune
```

This starts the real server at each candidate split, times a long-prompt request,
stops at the first value that will not allocate, and writes the winner to
`~/.local-claude/ncmoe`. It measures the actual server — same context, slots and
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
| 3072 / 4096 | — | compute buffer no longer fits |

**2.8× prefill, decode untouched.** The 512 default is sized for a model that
lives on the GPU. With experts on the CPU, prefill is bound by expert matrix
multiplies, and a larger micro-batch loads each expert once and reuses it across
more of the batch. `install.sh` sets this from your VRAM; raise it if you have
headroom, and re-check after changing the split.

## Everything else

| Env | Default | When to touch it |
|---|---|---|
| `LOCAL_CTX` | from VRAM (131072 on 8 GB) | Must exceed 100k, or Claude Code overruns the server before it auto-compacts |
| `LOCAL_NCMOE` | tuned value | Override the sweep |
| `LOCAL_UBATCH` | from VRAM | See above |
| `LOCAL_KV_TYPE` | `q8_0` | `f16` doubles KV memory for little quality gain |
| `LOCAL_SLOTS` | 2 | Slots share one KV pool; the second stops background calls evicting your conversation's prefix |
| `LOCAL_THREADS` | physical cores, capped at 16 | Memory bandwidth saturates around 8 threads; more rarely helps |
| `LOCAL_THINKING` | 1 | `0` disables chain-of-thought — much faster turns, worse reasoning |
| `LOCAL_MODEL_FILE` | best in `~/models` | Pick a model explicitly |
| `LOCAL_DEVICE` | autodetected | Force a GPU; never the integrated one |
| `LOCAL_TEMPLATE` | auto-patched | `0` uses the model's stock chat template — see DESIGN-NOTES |
| `LOCAL_SEARCH` | 1 | `0` drops the web-search MCP server |
| `LOCAL_RULES` | built-in | `""` disables the injected operating rules |
| `LOCAL_BACKEND` | `llamacpp` | `ollama` if you have one running |

## Sizing context

Context competes with expert layers for VRAM, but on a hybrid model it is cheap.
Qwen3.5/3.6-35B-A3B has real attention on only every fourth layer, with 2 KV
heads — about **11 KB per token** at `q8_0`, so a 131k window costs ~1.4 GB. A
conventional model of the same size wants roughly 16× that.

Do not shrink the window to buy expert layers without checking: Claude Code will
not auto-compact below 100k tokens, so a smaller pool means the server rejects a
long session rather than the client shrinking it.

## Still on the table

On the reference machine, decode sits ~10% below an ollama CUDA stack and prefill
trails it (621 vs ~1060 tok/s at matched prompt size). That gap is Vulkan versus
CUDA. Closing it means building llama.cpp with a CUDA toolkit and pointing
`LLAMA_DIR` at the result. Given that one micro-batch flag was worth 2.8×, it is
worth measuring rather than assuming.
