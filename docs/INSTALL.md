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

**NVIDIA users:** the installer chooses Vulkan deliberately, because llama.cpp
publishes no Linux CUDA binaries and Vulkan reaches the same matrix cores with
nothing installed. A CUDA build may still be faster — `bin/build-cuda.sh` does it:

```bash
./bin/build-cuda.sh          # builds and installs alongside; switches nothing
```

It needs `nvcc`, and this is the part that surprises people: **having "CUDA
installed" usually is not enough.** The NVIDIA driver gives you `libcuda` and
makes `nvidia-smi` report a CUDA version; PyTorch's pip wheels give you `cudart`
and `cuBLAS` with headers. Neither provides the offline compiler. Check with
`command -v nvcc` — if it is missing, install just the compiler rather than a
whole toolkit:

```bash
conda create -y -p ~/.local/share/local-claude/cudaenv -c conda-forge \
  cuda-nvcc cuda-cudart-dev cuda-cccl
```

(The PyPI `nvidia-cuda-nvcc-cu12` wheel is *not* sufficient — it ships `ptxas`
and `libnvvm` for runtime JIT, but not the `nvcc` driver binary.)

The script installs to `<prefix>/<build>-cuda` next to the Vulkan build and tells
you how to benchmark both. Adopt it only if it wins on your hardware:

```bash
ln -sfn ~/.local/share/local-claude/<build>-cuda ~/.local/share/local-claude/current
```

Already have a llama.cpp build of your own? Just point at it:

```bash
LLAMA_DIR=/path/to/llama.cpp/build/bin local-claude
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
