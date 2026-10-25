# Rebuild, September 2026: ollama → llama.cpp with MoE expert offload

What changed, why, and how to undo it.

## Why

The stack ran a 22 GB MoE model on an 8 GB card by letting ollama split the model
**by whole layers**: 10 of 40 layers on the GPU, 30 on the CPU. That spends VRAM on
expert weights that are read for roughly 3% of tokens, while evicting the attention
and KV cache that every token reads in full.

[An XDA article on MoE models and old GPUs](https://www.xda-developers.com/old-gpus-are-finally-practical-for-home-server-ai/)
describes the split that fits the architecture instead: keep the router, attention
and KV on the GPU, push the routed experts to system RAM, via llama.cpp's
`--n-cpu-moe`. ollama has no equivalent knob, so the backend moved to llama.cpp.

Two things came free with the move:

- `/v1/messages/count_tokens`, which ollama 404s — Claude Code was silently falling
  back to estimating its own context usage.
- Direct control of context, KV type, slots and prompt caching, which ollama only
  exposes as coarse environment variables.

## What changed

| Area | Before | After |
|---|---|---|
| Backend | ollama 0.31.2, `:11434` | llama.cpp `llama-server` b10941 (Vulkan), `:8080` |
| Model placement | whole-layer split, 25% GPU / 75% CPU | attention + KV + router + shared experts on GPU, routed experts in RAM (`--n-cpu-moe`) |
| Model | qwen3.5:35b (ollama library) | Qwen3.6-35B-A3B UD-Q4_K_XL (unsloth GGUF) — SWE-bench Verified 73.4 vs 70.0 |
| Context | 49152 | 131072 shared KV pool, 2 slots (`--kv-unified`) |
| Token counting | estimated by the client | exact, from the server |
| Model install | `ollama pull` | `local-claude pull` — resumable, sha256-verified, queueable |
| Tuning | none | `local-claude bench --tune` sweeps `--n-cpu-moe` against the real server |

New files: `bin/fetch-model.sh`, `bin/pull-after.sh`, `bin/auto-tune.sh`,
`bin/patch-gguf.py`, `docs/MOE-OFFLOAD.md`, `docs/MODELS.md`, `CLAUDE.md`.
Rewritten: `local-claude`, `test/bench.sh`, `test/api_compat_test.sh`.
Unchanged: `hooks/`, `mcp/`, `test/mcp_server_test.py`, `test/e2e_agentic_test.sh`
(retargeted only).

## What was tried and rejected

- **Reusing the 24 GB GGUF ollama had already downloaded.** The metadata mismatch
  is fixable (`bin/patch-gguf.py --fix-ollama` widens `rope.dimension_sections`
  from 3 to 4 elements, which is exactly what upstream GGUFs carry), but the
  tensors themselves are laid out differently — fused `attn_qkv`, `ssm_dt` without
  the `.bias` suffix, and `ssm.v_head_reordered = true`. Renaming would produce
  silent garbage. Details in [MODELS.md](MODELS.md).
- **A CUDA build.** llama.cpp ships no Linux CUDA binaries (Windows only) and this
  machine has no CUDA toolkit; installing one is a multi-GB download on a
  ~0.3 MB/s link. The Vulkan build uses the 4060's `NV_coopmat2` matrix cores and
  needs nothing installed.
- **GLM-4.7-Flash** as the model. Competitive size, but 20 KV heads with no GQA
  make its KV cache roughly 6× costlier per token, which would consume the VRAM
  the expert offload depends on.
- **Splitting bandwidth across two downloads.** Two concurrent pulls each ran at
  half speed; downloads are queued with `bin/pull-after.sh` instead.

## The one real integration bug

`api_compat_test.sh` passed 7/7 while Claude Code itself could not complete a
single turn. Claude Code sends its environment/context block as a **trailing
system message**, so the converted request is `system, user, system`; Qwen's stock
chat template raises `System message must be at the beginning.` and llama-server
returns HTTP 500. Hand-written requests never hit it because they put system
first.

The fix is `bin/make-template.sh`: it extracts the model's own template and
prepends a prelude that merges every system message into a single one at the
front. The launcher generates this per model into `~/.local-claude/templates/`
and passes `--chat-template-file`; `LOCAL_TEMPLATE=0` opts out.

Finding it took starting the server with `--verbose` and grepping the log for
`converted request:`, which prints the message list the template actually
receives — worth remembering for the next client-compatibility puzzle.

## Verified

- llama.cpp Vulkan inference on the RTX 4060 (device `Vulkan1`; `Vulkan0` is the
  Intel iGPU and must never be picked).
- `/v1/messages`, `/v1/messages/count_tokens` (HTTP 200, exact counts) and the full
  SSE event sequence — `message_start`, `content_block_start`, `content_block_delta`,
  `content_block_stop`, `message_delta`, `message_stop`.
- Response shape includes `usage.cache_read_input_tokens`, so Claude Code's prompt
  cache accounting is real rather than estimated.
- The ollama fallback (`LOCAL_BACKEND=ollama`) still starts, serves and answers.
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS` is now exported from the backend's real context
  size: Claude Code assumes a 200k window for any model it does not recognise,
  which is every local one, and would otherwise overrun the server's allocation
  before auto-compacting.
- Full agentic loop through Claude Code on llama.cpp: write a file, run it with
  Bash, verify the output, reply — **7 s** on the 2B plumbing model at the default
  131072 context.
- `api_compat_test.sh` 7/7 against llama-server (the ollama stack could only reach
  5/6, since it has no `count_tokens`).
- The target model's metadata was checked from its partial download before
  committing to it: `arch=qwen35moe` (supported), `rope.dimension_sections =
  [11, 11, 10, 0]` (no patching needed), 256 experts, `full_attention_interval = 4`,
  and a chat template with `tools`, `tool_call`, `tool_response` and
  `enable_thinking` support.

## Rolling back

The ollama install, its models and its launcher path are untouched:

```bash
LOCAL_BACKEND=ollama local-claude          # one session
export LOCAL_BACKEND=ollama                # or make it the default for a shell
```

To go back permanently, change `LOCAL_BACKEND` default at the top of
`local-claude`. Nothing else in the stack is llama.cpp-specific.

## Re-tune, October 2026

[A second XDA piece on `--n-cpu-moe` with 8 GB GPUs](https://www.xda-developers.com/i-stopped-thinking-my-8gb-gpu-was-too-small-after-learning-to-offload-moe-llm-layers-to-ram/)
prompted a re-check of the settings it does not cover. Adopted: one pinned thread per performance core,
and a micro-batch tuned with the split (numbers in [TUNING.md](TUNING.md)).
Tried and rejected:

- **Restricting threads to the P-cores without strict placement.** As fast when
  idle, but under background load the threads pile onto the same cores: decode
  fell to 14 tok/s with 8 busy processes, against 36.5 pinned.
- **Pinned expert memory by default** (`--load-mode none`). +7% prefill, but ~17 GB
  the kernel cannot reclaim on a machine that also runs experiments. Opt-in as
  `LOCAL_LOAD_MODE=none`.
- **The article's smaller quant (UD-IQ4_XS, 17.7 GB).** Not tried: ~13% fewer
  expert bytes would buy less than the two changes above, at a quality cost and a
  10–16 h download on this link.
