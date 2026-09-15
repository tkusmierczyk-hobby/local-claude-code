# Choosing and installing models

## The budget

| Resource | Reference machine | What lives there |
|---|---|---|
| VRAM | 8 GB (RTX 4060 Laptop) | attention, KV cache, router, shared experts, some expert layers |
| RAM | 31 GB total, ~25 GB usable | routed expert weights (mmapped from the GGUF) |
| Disk | 480 GB free | model files in `~/models` |

Practical ceiling: **a GGUF of about 22–24 GB**, and it must be a **MoE with a
small active parameter set**. A dense model of the same file size reads every
weight for every token, so the part that does not fit in VRAM becomes a hard
CPU-memory-bandwidth wall. See [MOE-OFFLOAD.md](MOE-OFFLOAD.md).

Two model properties matter beyond size:

1. **Active parameters** — 3 B active is the sweet spot; it is what makes CPU-side
   expert streaming fast enough.
2. **KV cache shape** — few KV heads, or a hybrid architecture where most layers
   have no KV cache at all. This decides how much context you can afford.

## Survey (September 2026)

| Model | Arch | Active | GGUF (Q4-class) | Verdict |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B** | MoE + Gated DeltaNet hybrid, 40 layers, 256 experts / 8 + 1 shared | 3 B | 22.4 GB (UD-Q4_K_XL) | **chosen** — SWE-bench Verified 73.4, LiveCodeBench v6 80.4, Terminal-Bench 2.0 51.5; 262k context; only 10 of 40 layers hold KV |
| Qwen3.5-35B-A3B | same family, previous generation | 3 B | 22–24 GB | what the old stack ran; SWE-bench Verified 70.0, Multilingual 60.3 vs 67.2 |
| GLM-4.7-Flash | MoE, 47 layers, 64 experts / 4 + 1 shared | ~3 B | 18.3 GB (Q4_K_M) | rejected: 20 KV heads with no GQA makes the KV cache ~6× costlier per token, which eats the VRAM the expert offload needs |
| Gemma-4-26B-A4B (QAT) | MoE | 4 B | 14.3 GB | viable lighter fallback if 22 GB is too much RAM pressure |
| Qwen3.8-27B | dense | 27 B | ~17 GB | wrong family for this box — dense means no sparse part to evict |
| Qwen3.5-122B-A10B | MoE | 10 B | ~70 GB+ | exceeds RAM |
| Qwen3.8-Flash-Next | MoE | — | 72 GB at IQ1_S | far beyond RAM |
| Ling-3.0-flash | MoE, 512 experts | ~6 B | — | too large, and 32 KV heads |
| Qwen3.5-2B | dense hybrid | 2 B | 1.3 GB | plumbing-test model only |

## Quantisation choice for Qwen3.6-35B-A3B

| Quant | Size | Fit |
|---|---|---|
| UD-Q5_K_M | 26.5 GB | too tight — experts alone would crowd out the page cache |
| **UD-Q4_K_XL** | **22.4 GB** | **chosen** — unsloth "dynamic" quant keeps sensitive tensors at higher precision |
| UD-Q4_K_S | 20.9 GB | a little more headroom, slightly worse |
| UD-IQ4_XS | 17.7 GB | good if you also want a large context and lots of free RAM |
| UD-Q3_K_XL | 16.9 GB | noticeably weaker at code; only if RAM is contested |
| UD-IQ2/IQ1 | 10–12 GB | not worth it for agentic coding |

## Installing a model

```bash
./local-claude pull unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450
```

`bin/fetch-model.sh` (what `pull` runs) is built for a slow link: it restarts curl
with `-C -` until the file is complete, so a suspend, a dropped connection or a
reboot only costs the current chunk. It verifies sha256 before moving the file into
place, and writes progress to `~/models/<file>.log`. Run it detached:

```bash
setsid nohup ./local-claude pull <repo> <file> <sha256> </dev/null >/dev/null 2>&1 &
./local-claude models          # progress of in-flight downloads
```

Downloads are queued, never run in parallel — two concurrent pulls each ran at
half speed and both finished later than they would have back to back:

```bash
setsid nohup bin/pull-after.sh ~/models/small.gguf <repo> <big-file.gguf> <sha256> \
  </dev/null >/dev/null 2>&1 &          # starts when small.gguf lands
setsid nohup bin/auto-tune.sh </dev/null >/dev/null 2>&1 &   # tunes when it finishes
```

A download that is running but averaging far below its instantaneous rate is the
signature of wasted re-transfers — check the log for repeated timeouts quoting the
same `out of <N> bytes` remaining total. `bin/fetch-model.sh` avoids the cause by
never passing curl `--retry`; see CLAUDE.md.

`local-claude models` marks a `.part` file **STALLED** when no fetcher process is
running for it, and prints the command to resume — expect to need that at least
once across a 20 h transfer. Re-running the pull is always safe: a complete `.part`
is verified and renamed, an incomplete one resumes. (One way to earn a STALLED: edit
`bin/fetch-model.sh` while it runs. bash reads scripts by byte offset, so the
running copy dies on the shifted content.)

A 22 GB model is a long download on a slow link — it took ~20 h at 0.3–0.5 MB/s on
the reference machine. Two things are worth checking before you wait it out: a VPN
often halves throughput (that was the limiter here), and a smaller quantisation
from the table above may be the better trade. The transfer is resumable either
way, so starting it and deciding later costs nothing.

## Vision

Qwen3.6-35B-A3B is multimodal, but the vision tower ships as a separate
**mmproj** file — without it the model is text-only. Claude Code sends images
when you paste a screenshot or `Read` an image file, so it is worth having.

```bash
local-claude pull unsloth/Qwen3.6-35B-A3B-GGUF mmproj-F16.gguf \
  8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f
```

0.90 GB. Once it is in `~/models`, the launcher finds it automatically — there is
nothing to configure. `local-claude status` and the startup line show `+vision`
when it is active. With more than one mmproj present the launcher cannot tell
which model each belongs to, so set `LOCAL_MMPROJ` explicitly.

Three things worth knowing:

- **The projector runs on the CPU by default here.** It wants ~1 GB of VRAM,
  which is exactly what the expert layers are competing for, and an image is
  encoded once per message rather than per token — so the GPU is the wrong place
  for it on a card this size. `LOCAL_MMPROJ_DEVICE=gpu` overrides that if you
  have headroom.
- **Images cost context.** Each one becomes hundreds to thousands of tokens.
  `--image-min-tokens` / `--image-max-tokens` bound that; pass them through
  `LOCAL_LLAMA_ARGS` if you need to.
- **F16 is the one to take.** BF16 is the same size with no advantage here, and
  F32 doubles the file for precision the vision tower does not need.

If you would rather stay text-only, simply do not download it — or move it out of
`~/models`.

## Switching models

The launcher picks the best model in `~/models` automatically, preferring
`Qwen3.6-35B-A3B*`, then any `*35B-A3B*`, then any `*A3B*`, then the newest GGUF.
Half-downloaded `.part` files are ignored. To override:

```bash
LOCAL_MODEL_FILE=~/models/Qwen3.5-2B-Q4_K_M.gguf ./local-claude
```

If the server is already running with a different model, the launcher stops and
restarts it. After switching, re-tune the expert split:

```bash
./local-claude bench --tune
```

## Gotcha: ollama blobs are not reusable

An obvious-looking shortcut is to point llama.cpp at the 24 GB GGUF ollama already
downloaded (`~/.ollama/models/blobs/sha256-*`). It does not work, and it is worth
knowing why before you try:

1. **Metadata** — ollama writes `qwen35moe.rope.dimension_sections` with 3 elements;
   llama.cpp reads it into a 4-element array and hard-fails. This part is fixable:
   `bin/patch-gguf.py IN OUT --fix-ollama` rewrites the key (upstream GGUFs carry
   `[11, 11, 10, 0]`) while streaming the tensor data through unchanged.
2. **Tensor layout** — after that fix it still fails on
   `tensor 'blk.0.ssm_dt.bias' not found`. ollama's converter emits a different
   dialect: fused `attn_qkv` where llama.cpp wants separate projections, `ssm_dt`
   without the `.bias` suffix, and a `ssm.v_head_reordered = true` flag meaning the
   V heads are physically permuted. That is a data-layout difference, not a naming
   one, so renaming tensors would silently produce garbage.

Download the GGUF built for llama.cpp (unsloth, bartowski, ggml-org) instead.
`bin/patch-gguf.py` is still useful for inspecting any GGUF: `--show` dumps the
full metadata block.
