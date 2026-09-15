# Installing

```bash
./install.sh
```

That is the whole thing. It detects your hardware, picks a matching llama.cpp
build, installs it in user space, generates the per-machine configuration, links
the launcher onto your `PATH`, and runs a preflight check. It needs no root and
touches nothing outside your home directory.

## What goes where

| Path | Contents |
|---|---|
| `~/.local/share/local-claude/` | llama.cpp builds, with `current` symlinked to the active one |
| `~/.local-claude/` | state: tuned split, detected device, chat templates, logs, MCP config |
| `~/models/` | GGUF model files |
| `~/.local/bin/local-claude` | symlink to the launcher in this repo |

The repo itself stays wherever you cloned it; the symlink resolves back to it, so
`git pull` updates the running stack.

## Options

```bash
./install.sh --backend cpu      # force a backend: vulkan | cpu | metal
./install.sh --build b10988     # pin a llama.cpp build instead of the newest
./install.sh --no-symlink       # don't touch ~/.local/bin
```

Environment overrides: `LOCAL_CLAUDE_PREFIX`, `LOCAL_CLAUDE_STATE`, `MODELS_DIR`.

## Platforms

| Platform | Backend | Status |
|---|---|---|
| Linux x64 + NVIDIA | Vulkan | tested — the reference machine |
| Linux x64, no GPU | CPU | supported; only small models are practical |
| Linux arm64 | CPU | supported, untested |
| macOS arm64 | Metal | supported, untested — unified memory means VRAM sizing differs |

**NVIDIA users:** the installer chooses Vulkan deliberately. llama.cpp publishes
no Linux CUDA binaries, and building from source needs a multi-gigabyte toolkit.
Vulkan reaches the same matrix cores with nothing installed. If you already have
CUDA and want the extra speed, build llama.cpp yourself and point the stack at it:

```bash
LLAMA_DIR=/path/to/your/llama.cpp/build/bin local-claude
```

## Picking a llama.cpp build

llama.cpp publishes two kinds of release: version tags (`v0.4.1`) that carry no
binaries, and build tags (`b10988`) that do. `releases/latest` returns the former,
so the installer walks recent releases and takes the newest that actually has a
binary for your platform. If you need to pin one, pass `--build`.

The stack requires a build new enough to have **`--n-cpu-moe`**; preflight checks
this explicitly and fails loudly if it is missing.

## Verifying

```bash
./test/preflight.sh
```

Checks the binary runs, `--n-cpu-moe` exists, a discrete GPU is visible, RAM and
disk are sufficient, the MCP config was generated, the `claude` CLI is present,
and the port is free. Run it any time something looks wrong.

## Updating

```bash
git pull
./install.sh          # picks up a newer llama.cpp; keeps your models and tuning
```

Old builds are left in place, so rolling back is repointing the symlink:

```bash
ln -sfn ~/.local/share/local-claude/b10941 ~/.local/share/local-claude/current
```

After changing builds, re-run `local-claude bench --tune` — kernels change, and
so does the best expert split.

## Uninstalling

```bash
rm -rf ~/.local/share/local-claude ~/.local-claude ~/.local/bin/local-claude
rm -rf ~/models        # only if you want the model files gone too
```
