#!/usr/bin/env bash
# detect-hardware.sh — describe this machine, and what it should run.
#
# Prints shell-evaluable KEY=VALUE lines so both the installer and the launcher
# can share one view of the hardware:
#
#   eval "$(bin/detect-hardware.sh)"
#
# Everything it reports is a starting point, not a verdict. The only figure that
# really matters — how many expert layers fit in VRAM — is measured later by
# `local-claude bench --tune`, because it depends on the model, the quantisation,
# the context size and the KV cache type all at once.
set -uo pipefail

kv() { printf '%s=%s\n' "$1" "$(printf '%q' "$2")"; }

# ---- platform ----------------------------------------------------------------
case "$(uname -s)" in
  Linux)  OS=linux ;;
  Darwin) OS=macos ;;
  *)      OS=unknown ;;
esac
case "$(uname -m)" in
  x86_64|amd64) ARCH=x64 ;;
  arm64|aarch64) ARCH=arm64 ;;
  *) ARCH=unknown ;;
esac

# ---- memory and cores --------------------------------------------------------
if [ "$OS" = macos ]; then
  RAM_MB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576 ))
  CORES=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 4)
else
  RAM_MB=$(awk '/^MemTotal:/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)
  # Physical cores, not hyperthreads: memory bandwidth saturates well before
  # thread count does, and oversubscribing costs more than it gains.
  CORES=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{gsub(/ /,"",c); gsub(/ /,"",s); print (c*s>0)? c*s : 0}')
  [ "${CORES:-0}" -gt 0 ] 2>/dev/null || CORES=$(( $(nproc --all 2>/dev/null || echo 4) / 2 ))
fi
THREADS=$(( CORES > 16 ? 16 : CORES ))
[ "$THREADS" -lt 1 ] && THREADS=1

# ---- GPU ---------------------------------------------------------------------
# Checked one path at a time: `ls a b` fails whenever *any* argument is missing,
# which silently reported "no Vulkan" on a machine that had it.
have_vulkan() {
  local p
  for p in /usr/lib/x86_64-linux-gnu/libvulkan.so.1 /usr/lib64/libvulkan.so.1 \
           /usr/lib/aarch64-linux-gnu/libvulkan.so.1 /usr/lib/libvulkan.so.1 /usr/lib/*/libvulkan.so.1; do
    [ -e "$p" ] && return 0
  done
  return 1
}

GPU_NAME="none"; VRAM_MB=0; BACKEND=cpu
if [ "$OS" = macos ] && [ "$ARCH" = arm64 ]; then
  GPU_NAME="Apple Silicon (unified memory)"
  VRAM_MB=$(( RAM_MB * 2 / 3 ))   # unified: the GPU may address most of RAM
  BACKEND=metal
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  # Vulkan is preferred over CUDA here only because llama.cpp publishes no Linux
  # CUDA binaries; a CUDA build from source is faster if you have the toolchain.
  if have_vulkan; then BACKEND=vulkan; else BACKEND=cpu; fi
elif have_vulkan && command -v vulkaninfo >/dev/null 2>&1; then
  GPU_NAME=$(vulkaninfo --summary 2>/dev/null | awk -F= '/deviceName/{print $2; exit}' | sed 's/^ *//')
  BACKEND=vulkan
fi
[ -z "${VRAM_MB:-}" ] && VRAM_MB=0

# ---- what to run -------------------------------------------------------------
# Experts are memory-mapped from the GGUF, so the model must fit in RAM with room
# left for the OS and the page cache. Anything larger thrashes the disk per token.
USABLE_RAM_MB=$(( RAM_MB - 7000 ))
[ "$USABLE_RAM_MB" -lt 0 ] && USABLE_RAM_MB=0

if   [ "$USABLE_RAM_MB" -ge 23000 ]; then REC_TIER=full
elif [ "$USABLE_RAM_MB" -ge 17000 ]; then REC_TIER=reduced
elif [ "$USABLE_RAM_MB" -ge 14000 ]; then REC_TIER=small
else                                      REC_TIER=tiny
fi

# Context and micro-batch both consume VRAM that expert layers want. These are
# conservative starting points; bench --tune re-checks them against reality.
if   [ "$VRAM_MB" -ge 15000 ]; then REC_CTX=262144; REC_UBATCH=2048
elif [ "$VRAM_MB" -ge 7000  ]; then REC_CTX=131072; REC_UBATCH=2048
elif [ "$VRAM_MB" -ge 5000  ]; then REC_CTX=65536;  REC_UBATCH=1024
elif [ "$VRAM_MB" -ge 3000  ]; then REC_CTX=32768;  REC_UBATCH=512
else                                REC_CTX=32768;  REC_UBATCH=512
fi

kv HW_OS "$OS"
kv HW_ARCH "$ARCH"
kv HW_BACKEND "$BACKEND"
kv HW_GPU_NAME "$GPU_NAME"
kv HW_VRAM_MB "$VRAM_MB"
kv HW_RAM_MB "$RAM_MB"
kv HW_CORES "$CORES"
kv HW_THREADS "$THREADS"
kv REC_TIER "$REC_TIER"
kv REC_CTX "$REC_CTX"
kv REC_UBATCH "$REC_UBATCH"
kv REC_USABLE_RAM_MB "$USABLE_RAM_MB"
