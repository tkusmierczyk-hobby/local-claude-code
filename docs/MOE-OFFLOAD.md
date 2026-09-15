# MoE expert offload: putting the right 8 GB on the GPU

This is the technique the stack is built on, why it beats what we did before,
and how to re-tune it when the hardware or the model changes.

All measurements below come from the **reference machine**: an RTX 4060 Laptop
(8 GB VRAM), i9-13900HX, 31 GB of DDR5-5600, Vulkan backend. Your numbers will
differ; the reasoning transfers, and `local-claude bench --tune` re-measures the
one value that matters.

## The problem

An RTX 4060 Laptop has 8 GB of VRAM. The model we want to run is 22 GB. Something
has to live in system RAM, and *what* you choose to evict decides whether you get
30 tokens/s or 3.

The old stack (ollama) made that choice by **whole layers**: fit as many complete
transformer layers into VRAM as possible — attention *and* feed-forward together —
and run the rest on the CPU. On the reference machine that was a 25 / 75 split: 10 of 40
layers on the GPU, 30 layers on the CPU.

## Why whole-layer splitting is the wrong split for MoE

A Mixture-of-Experts layer is not one feed-forward network, it is 256 small ones
plus a router that picks 8 of them per token. In Qwen3.6-35B-A3B:

| Part of the model | Share of weights | Read per token |
|---|---|---|
| Routed experts | ~90% | 8 of 256 (~3%) |
| Attention, router, shared expert, norms, embeddings | ~10% | all of it |
| KV cache | grows with context | all of it, every token |

So when a whole layer goes to VRAM, most of the VRAM it consumes is expert weights
that a given token will never touch. Meanwhile attention and the KV cache — which
*every* token reads in full — get evicted to RAM for 30 of 40 layers.

The fix is to split by **tensor role** instead of by layer:

- **GPU**: attention, KV cache, router, shared experts, norms — small, hot, and
  latency-critical.
- **RAM**: routed expert weights — bulky, and only ~3% of them are read per token,
  so the CPU only has to stream ~3 B of parameters per token instead of 35 B.

This is what makes a 35B model practical on an 8 GB card, and it is why MoE models
(not dense ones) are the right family for this hardware. A dense 27B has no sparse
part to evict: every weight is read for every token, so whatever does not fit in
VRAM is pure CPU-bandwidth cost.

## The flags

llama.cpp implements the split as a tensor-buffer override:

| Flag | Meaning |
|---|---|
| `-ncmoe, --n-cpu-moe N` | keep the routed experts of the **first N layers** in system RAM |
| `-cmoe, --cpu-moe` | keep **all** routed experts in system RAM (equivalent to N = n_layers) |
| `-ot, --override-tensor <regex>=<buffer>` | the general mechanism both of the above are shorthand for |
| `-ncffn, --n-cpu-ffn N` | the same idea for dense models (no experts to be sparse about) |

Combine with `-ngl 999` — "put everything on the GPU" — and the expert override
then pulls just the experts back out. Ordering matters conceptually: `-ngl` decides
what is offloaded, `-ncmoe` carves the experts out of that.

The stack sets this in `local-claude`; `LOCAL_NCMOE` overrides it per run.

## Tuning it

Lower N = fewer expert layers in RAM = more expert weights in VRAM = faster —
right up until VRAM overflows, at which point it collapses (or fails to allocate).
The tuning job is to find the smallest N that still fits, and it is a sweep:

```bash
./local-claude bench --tune              # sweep N, store the winner
./local-claude bench --tune --sweep 40,36,32,28
```

`bench --tune` walks N downward, and at each value starts the **real
llama-server** through `local-claude start` and times one long-prompt request
(~4k tokens of prefill, then 128 tokens of generation). It stops at the first
value where the server cannot allocate, and writes the best to
`~/.local-claude/ncmoe`, which the launcher then uses.

Tuning against the real server rather than `llama-bench` matters: context size,
slot count and KV cache type all take VRAM away from the expert layers, and
`llama-bench` sizes its KV cache from `-p/-n` rather than from `--ctx-size`, so it
would happily bless a split that the server cannot actually load. Re-run it after changing
model, quantisation, context size, or KV cache type — all four move the VRAM budget.

Rules of thumb when reading the sweep:

- **Decode (tok/s)** is what you feel while the model types. It is bound by how
  fast expert weights stream from RAM, so it improves slowly as N drops.
- **Prefill (tok/s)** is what you wait for when Claude Code sends a 15k-token
  system prompt. It is compute-bound and improves sharply when more work lands on
  the GPU.
- If a value fails to allocate, you have found the ceiling; back off by 2.

## Why context is cheap on this model

Qwen3.5/3.6-35B-A3B is a **hybrid**: `full_attention_interval = 4` means only every
4th layer is a real attention layer with a KV cache (10 of 40); the other 30 are
Gated DeltaNet (linear attention), which carries a fixed-size recurrent state
instead of a per-token cache. Those 10 layers use grouped-query attention with just
2 KV heads of 256 dims.

With `q8_0` KV that works out to roughly **11 KB per token**:

```
10 layers × 2 kv heads × 256 dims × 2 (K and V) × 1.0625 B/value ≈ 10.9 KB/token
65536 tokens ≈ 0.7 GB     131072 tokens ≈ 1.4 GB
```

A conventional 40-layer MHA model of the same size would want 6–10× that. That is
why the stack runs a **131072-token** pool on an 8 GB card without giving up many
expert layers. The size is not arbitrary: Claude Code's `--autocompact` floor is
100k tokens, so a smaller pool would let a long session overflow the server before
the client ever decided to compact.

## Measured

Qwen3.6-35B-A3B UD-Q4_K_XL, llama.cpp b10941 Vulkan, 2026-09-15. Full data in
`~/.local-claude/bench-results.txt`.

| `--n-cpu-moe` | prefill tok/s | decode tok/s | VRAM |
|---|---|---|---|
| 40 (all experts in RAM) | 159.0 | 26.8 | 4115 MiB |
| 38 | 189.2 | 27.7 | 5111 MiB |
| 36 | 198.6 | 28.4 | 6039 MiB |
| **34** | 210.5 | **29.5** | 6967 MiB |
| 32 | — | — | did not start |

Each pair of layers moved onto the GPU costs ~930 MiB, which matches the predicted
488 MB per layer of expert bank almost exactly. The VRAM arithmetic in this
document is reliable; the throughput arithmetic below is not, and it is worth
being precise about how.

## Where the ceiling model was wrong

This document predicted a ~80 tok/s ceiling for the tuned split, and reasoned that
since the previous stack achieved 71% of *its* ceiling, this one would land near
55 tok/s. It lands at **29.5**, about 37% of the bound.

The model was right about everything comparative — the ranking of configurations,
the VRAM cost per layer, the direction and rough size of each improvement — and
wrong about the absolute number, because a pure bandwidth bound ignores three
costs that a CPU-side expert path actually pays:

1. **Dequantisation.** Q4_K blocks are unpacked before they can be multiplied.
   That is CPU work proportional to the bytes read, and it appears nowhere in a
   "bytes ÷ bandwidth" calculation.
2. **Synchronisation.** Placement splits the model at 40 layer boundaries, so a
   token involves ~80 handoffs between devices. Each is cheap; none are free.
3. **Batch-one GEMMs.** At decode the expert multiply is matrix-by-vector, which
   is latency-sensitive rather than bandwidth-saturating; the CPU does not reach
   the 65 GB/s a streaming benchmark reports.

The lesson is not that the arithmetic was useless — it chose the right strategy
and predicted the memory layout correctly. It is that a bandwidth bound is an
upper bound on the *memory* term only, and should be quoted as "no faster than",
never converted into an expected figure by borrowing an efficiency ratio measured
on a differently-shaped workload.

## The flag that mattered more than the tuning

With the split fixed at `--n-cpu-moe 34`, varying the micro-batch size:

| `--ubatch-size` | prefill tok/s | decode tok/s |
|---|---|---|
| 512 (llama.cpp default) | 231.9 | 30.5 |
| 1024 | 391.8 | 31.6 |
| **2048** | **648.1** | 31.0 |
| 3072 / 4096 | — | did not start (compute buffer too large) |

**2.8&times; prefill for one flag**, and decode is untouched. The default of 512 is
sized for a model that lives on the GPU. When the experts are on the CPU, prefill
is bound by expert GEMM throughput, and a larger micro-batch means each expert is
loaded once and reused across far more tokens in the batch — the same reuse
argument that makes prefill compute-bound in the first place, applied to the CPU
side. Above 2048 the compute buffers no longer fit in what the expert layers left
behind, which is the trade-off to re-check on a card with more VRAM.

In production (`--n-cpu-moe 34 --ubatch-size 2048 --ctx-size 131072`): prefill
**760.7 tok/s** on a 14k-token prompt, decode **28.7 tok/s**, 7396 MiB of VRAM,
API conformance 7/7, and the agentic end-to-end test passing in **60 s** against
the previous stack's 76 s.

## Backend, revisited

This document previously listed the Vulkan-versus-CUDA gap as an open question,
on the grounds that it was worth measuring rather than assuming. It was measured:
a source-built CUDA backend gives **+23% decode and +17% prefill** at the same
split, and the agentic end-to-end test drops from 60 s to 40 s. Decode at 34.6
tok/s now exceeds the old whole-layer ollama stack's 32.1, while running a newer
model at 2.7x the context.

The placement argument is unaffected — both backends want the same split, and the
expert offload is what makes either of them viable. See [TUNING.md](TUNING.md) for
the numbers and `bin/build-cuda.sh` for the build.
