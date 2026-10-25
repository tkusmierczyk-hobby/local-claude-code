# Contributing

## Before you send a change

```bash
./test/preflight.sh          # install sanity, no model needed
./test/api_compat_test.sh    # the contract with Claude Code — run after any launcher change
./test/e2e_agentic_test.sh   # the real loop; slower, but it catches what the others miss
./test/mcp_server_test.py
bash -n local-claude install.sh bin/*.sh test/*.sh
```

`api_compat_test.sh` passing is necessary but **not sufficient**: it once passed
7/7 while Claude Code itself could not complete a single turn, because a real
client sends message shapes curl does not. Run the end-to-end test too.

## Ground rules

1. **User space only.** No `sudo`, nothing outside `$HOME`. People run this on
   machines they do not administer.
2. **Nothing machine-specific in the repo.** Paths, tuned values, device choices
   and generated configs live in `~/.local-claude`. `install.sh` writes them.
3. **Inference stays local, and nothing contacts Anthropic.** The only network use
   is model downloads and the opt-out web-search tool; Claude Code's own web tools
   stay disabled.
4. **Validate llama.cpp flags** against `llama-server --help` before shipping
   them — a bad flag means the server refuses to start, with a confusing error.
5. **Measure claims.** Throughput numbers in docs should come from
   `bench --tune` output, and say which machine produced them.

## Reporting a performance result

Useful reports include: GPU and VRAM, total RAM, model and quantisation, the
output of `local-claude bench --tune`, and the `--ubatch-size` you ended up with.
That last one varies more than people expect.

## Notes for agents

[CLAUDE.md](CLAUDE.md) carries the invariants and the accumulated gotchas —
read it before changing the launcher or the download path. Several of those
entries cost hours to learn.
