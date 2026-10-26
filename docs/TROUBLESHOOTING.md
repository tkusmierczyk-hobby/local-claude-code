# Troubleshooting

Start with `./test/preflight.sh`. It catches most installation problems without
needing a model loaded.

## The server will not start

**"did not start (out of VRAM)"** — the expert split is too aggressive for your
card. Delete the tuned value and re-measure:

```bash
rm -f ~/.local-claude/ncmoe ~/.local-claude/ncmoe.* ~/.local-claude/ubatch.* && local-claude bench --tune
```

**Something else holds the GPU.** A previous server, or a benchmark run, may still
own the VRAM. `local-claude stop`, then check with `nvidia-smi`. A smaller
neighbour, such as an idle PyTorch process, is enough to push out a tuned 4096
micro-batch; the launcher then prints `retrying at 2048` and starts anyway, with
slower prefill.

**It started, then died mid-session.** If `~/.local-claude/llama-server.log` ends
with `CUDA error: out of memory` (often in `cudaGraphInstantiate`), the split left
too little VRAM for what the server allocates while running. Re-run
`local-claude bench --tune`, which keeps a reserve (`LOCAL_VRAM_MARGIN`, 768 MiB);
raise the reserve if it happens again. `GGML_CUDA_DISABLE_GRAPHS=1 local-claude`
also avoids it, at about 4% of decode speed.

**It picked the wrong GPU.** Machines with integrated graphics usually enumerate
them first. The launcher caches its choice:

```bash
cat ~/.local-claude/device          # e.g. Vulkan1
rm ~/.local-claude/device           # re-detect
LOCAL_DEVICE=Vulkan1 local-claude   # or force it
```

## Every request fails with an HTTP 500 Jinja error

If it says **"System message must be at the beginning"**, the model's stock chat
template is rejecting Claude Code's payload, which puts a system message last. The
launcher normally patches around this automatically. Force a regeneration:

```bash
rm ~/.local-claude/templates/<model>.jinja
local-claude stop && local-claude start
```

`LOCAL_TEMPLATE=0` reverts to the stock template, which will fail again — it is
for diagnosis, not a fix. Background in [DESIGN-NOTES.md](DESIGN-NOTES.md).

## Replies are empty, or contain only reasoning

Thinking models spend the output budget on reasoning first. If `max_tokens` is
small, the whole budget goes to the `thinking` block and no text comes out. Raise
`CLAUDE_CODE_MAX_OUTPUT_TOKENS`, or turn reasoning off:

```bash
LOCAL_THINKING=0 local-claude
```

## "Error writing file" / InputValidationError: could not be parsed as JSON

The model was cut off mid-tool-call. A `Write` of a long document is a single JSON
string; when the output budget runs out partway through it, the call arrives with
an unterminated string and Claude Code cannot parse it. The giveaway is a failed
call whose size is suspiciously close to your token cap — 32 KB of prose is about
8k tokens.

Two things cause it, and the launcher now handles both:

- **The cap was too low.** `CLAUDE_CODE_MAX_OUTPUT_TOKENS` defaults to 32768 here,
  not Claude Code's smaller default. Thinking tokens come out of the *same* budget,
  so leave headroom.
- **The model wrote too much at once.** The injected rules tell it to create a file
  with the first section and append the rest, rather than emitting hundreds of
  lines in one call.

If you still hit it, raise the cap further or disable reasoning for that task:

```bash
CLAUDE_CODE_MAX_OUTPUT_TOKENS=65536 local-claude
LOCAL_THINKING=0 local-claude        # frees the whole budget for output
```

Note that a long generation is genuinely slow — 32k tokens at 30 tok/s is ~18
minutes — so `API_TIMEOUT_MS` must stay large enough not to abort it.

## The session overflows or "autocompact is thrashing"

Claude Code assumes a 200k window for any model it does not recognise. The
launcher tells it the truth by reading the running server's real context and
exporting `CLAUDE_CODE_MAX_CONTEXT_TOKENS`. If you see this anyway, your context
is too small — it must clear 100k, because Claude Code will not compact below
that. Raise `LOCAL_CTX`, then re-tune.

## Downloads

**Stalled.** `local-claude models` labels a `.part` file `STALLED` when no fetcher
is running for it, and prints the resume command. Re-running a pull is always
safe: a complete file is verified and renamed, an incomplete one resumes.

**Slow, or slower than the speed it reports.** If the average is far below every
instantaneous sample, transfers are being re-downloaded and discarded. The fetcher
avoids the known cause (never passing curl `--retry` alongside `-C -`), but a VPN
or a throttling ISP will do it too.

**Never edit `bin/*.sh` while a download runs.** bash reads a script by byte
offset, so an edit shifts the remaining bytes under the running shell and it dies
mid-file. Re-run the pull to resume.

## It is slower than you expected

Run `local-claude bench --tune` if you have not. Then check `--ubatch-size` — the
default of 512 costs about 2.8× prefill on this kind of setup. See
[TUNING.md](TUNING.md).

If decode is far below the table there, confirm the model actually fits in RAM.
Experts are memory-mapped; if the file is larger than free RAM, every token pages
from disk and throughput collapses. `local-claude status` shows both.

## Web search does not work

It needs `requests` and `beautifulsoup4` for the python that `install.sh` found:

```bash
python3 -m pip install requests beautifulsoup4
./install.sh                 # regenerates the MCP config
```

DuckDuckGo rate-limits and intermittently refuses, and after a burst of searches
it answers with a bot check for a while. The search tool then tells the model it
is blocked, rather than that nothing was found, and the test reports it as SKIP
rather than failure. Waiting a few minutes clears it. `LOCAL_SEARCH=0` disables
the server entirely.

## "the running server requires an API key"

The server was started with `LOCAL_API_KEY` and this session has none (or a
different one). Start the session with the same `LOCAL_API_KEY`, or
`local-claude stop` and start without a key. The reverse — a key set for a
server started without one — only prints a note: the server is unprotected until
restarted.

## `/resume` does not list a session

Local sessions keep their history in `~/.local-claude/claude`, apart from your
normal Claude Code sessions, so each kind only lists its own. Sessions from before
the separation stay in `~/.claude`. With `LOCAL_STORAGE=none` nothing is kept.

## The first local session asks for a theme and whether to trust the folder

Expected once: local sessions have their own Claude Code settings, so the
first-run screens appear again, and each project folder is trusted once more.

## Seeing what the model reads and writes

From the most convenient to the most raw:

1. **In a session:** Ctrl+O switches Claude Code to its detailed view — the
   model's thinking, every tool call and the full tool results.
2. **After a session:** `~/.local-claude/claude/projects/<folder>/<session>.jsonl`
   holds every message as JSON (with the default `LOCAL_STORAGE=separate`). It does
   not include Claude Code's system prompt or tool definitions.
3. **The exact requests and replies:** restart the server with verbose logging.

   ```bash
   local-claude stop
   LOCAL_LLAMA_ARGS="--verbose" local-claude start
   grep 'converted request' ~/.local-claude/llama-server.log   # full input: system prompt, tools, messages
   grep 'Parsed message' ~/.local-claude/llama-server.log      # each finished reply: text, thinking, tool calls
   ```

   This writes every prompt to the log. Empty it when you are done:
   `: > ~/.local-claude/llama-server.log`.
4. **The raw text, as the model sees it:** render a conversation through the
   chat template, then send that text and read the output before any parsing —
   the control tokens (`<|im_start|>`), the thinking and the tool-call markup
   (`<tool_call><function=…>`) all show.

   ```bash
   curl -s http://127.0.0.1:8080/apply-template -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Hi"}]}'          # -> {"prompt": "..."}
   curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' \
     -d '{"prompt": "<the prompt from above>", "n_predict": 2000}'
   ```

   Pass tool definitions as `"tools": [...]` to `/apply-template` to see how the
   template presents them. Leave `n_predict` generous: the thinking comes first.
   With `LOCAL_API_KEY` set, add `-H "Authorization: Bearer <key>"`.

## Claude Code says `unrecognized_model`

Expected, and cosmetic. Local models are not in the client's catalog. The context
warning that used to accompany it is handled by the launcher.
