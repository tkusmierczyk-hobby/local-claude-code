#!/usr/bin/env bash
# build-cuda.sh — build llama.cpp with CUDA, for NVIDIA users who want the
# last bit of prefill speed that the prebuilt Vulkan binaries leave behind.
#
# llama.cpp publishes no Linux CUDA binaries, so this is a source build. It needs
# nvcc, which the NVIDIA *driver* does not provide — the driver gives you
# libcuda, PyTorch's pip wheels give you cudart and cuBLAS, but the offline
# compiler comes only from a toolkit.
#
#   ./bin/build-cuda.sh                  # find a toolkit, build, install alongside
#   ./bin/build-cuda.sh --install-nvcc   # fetch a compiler first (~185 MB, via conda)
#   ./bin/build-cuda.sh --nvcc /path/to/nvcc
#   ./bin/build-cuda.sh --build b10988   # pin the llama.cpp source revision
#   ./bin/build-cuda.sh --arch 89        # compute capability, if detection fails
#
# The result is installed next to the Vulkan build rather than over it, so you can
# measure both before choosing. Nothing is switched automatically.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${LOCAL_CLAUDE_PREFIX:-$HOME/.local/share/local-claude}"
NVCC=""; SRC_BUILD=""; ARCH=""; JOBS="$(nproc 2>/dev/null || echo 4)"; INSTALL_NVCC=0

while [ $# -gt 0 ]; do
  case "$1" in
    --nvcc)  NVCC="$2"; shift 2 ;;
    --install-nvcc) INSTALL_NVCC=1; shift ;;
    --build) SRC_BUILD="$2"; shift 2 ;;
    --arch)  ARCH="$2"; shift 2 ;;
    --jobs)  JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die() { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. find the pieces of a toolkit ----------------------------------------
say "1/5  Locating a CUDA toolchain"
find_nvcc() {
  local cand
  for cand in "$PREFIX/cudaenv/bin/nvcc" /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc \
              "$(command -v nvcc 2>/dev/null || true)"; do
    [ -n "$cand" ] && [ -x "$cand" ] && { echo "$cand"; return 0; }
  done
  return 1
}
[ -z "$NVCC" ] && NVCC="$(find_nvcc || true)"

if [ -z "$NVCC" ] && [ "$INSTALL_NVCC" = 1 ]; then
  command -v conda >/dev/null 2>&1 \
    || die "--install-nvcc needs conda on PATH. Install a toolkit another way and pass --nvcc"
  info "installing a compiler into $PREFIX/cudaenv (~185 MB, isolated — your base env is untouched)"
  conda create -y -p "$PREFIX/cudaenv" -c conda-forge cuda-nvcc cuda-cudart-dev cuda-cccl \
    || die "conda install failed"
  NVCC="$(find_nvcc || true)"
fi
[ -n "$NVCC" ] && [ -x "$NVCC" ] || die "no nvcc found.
  Having \"CUDA installed\" is usually not enough: the driver provides libcuda and
  makes nvidia-smi report a CUDA version, and PyTorch's wheels provide cudart and
  cuBLAS — none of them provide the compiler. Get one with:
    ./bin/build-cuda.sh --install-nvcc      (~185 MB, into $PREFIX/cudaenv)
  or pass --nvcc /path/to/nvcc if you have a toolkit elsewhere."
info "nvcc      $NVCC"
info "version   $("$NVCC" --version | tail -1)"

for t in cmake ninja; do
  command -v "$t" >/dev/null 2>&1 || {
    for p in "$PREFIX/buildenv/bin/$t" "$PREFIX/cudaenv/bin/$t"; do
      [ -x "$p" ] && { export PATH="$(dirname "$p"):$PATH"; break; }
    done
  }
  command -v "$t" >/dev/null 2>&1 || die "missing build tool: $t
  Install with: python3 -m venv $PREFIX/buildenv && $PREFIX/buildenv/bin/pip install cmake ninja"
done
info "cmake     $(cmake --version | head -1)"

if [ -z "$ARCH" ]; then
  ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
fi
[ -n "$ARCH" ] || die "could not detect compute capability; pass --arch (e.g. 89 for Ada, 86 for Ampere)"
info "arch      sm_$ARCH"

# ---- 2. source ---------------------------------------------------------------
say "2/5  Fetching llama.cpp source"
if [ -z "$SRC_BUILD" ]; then
  # Match the installed binaries so the CUDA build is a like-for-like swap.
  SRC_BUILD="$(basename "$(readlink -f "$PREFIX/current" 2>/dev/null)" 2>/dev/null || true)"
  case "$SRC_BUILD" in b[0-9]*|v[0-9]*) ;; *) SRC_BUILD="" ;; esac
fi
[ -n "$SRC_BUILD" ] || SRC_BUILD="$(curl -fsSL 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10' \
  | python3 -c 'import sys,json;print(next(r["tag_name"] for r in json.load(sys.stdin) if r.get("assets")))')"
info "revision  $SRC_BUILD"

SRC="$PREFIX/src/llama.cpp-$SRC_BUILD"
if [ ! -f "$SRC/CMakeLists.txt" ]; then
  mkdir -p "$PREFIX/src"
  TARBALL="$PREFIX/src/$SRC_BUILD.tar.gz"
  [ -s "$TARBALL" ] || curl -fL -C - --progress-bar -o "$TARBALL" \
    "https://github.com/ggml-org/llama.cpp/archive/refs/tags/${SRC_BUILD}.tar.gz" \
    || die "could not download llama.cpp source for $SRC_BUILD"
  rm -rf "$SRC"; mkdir -p "$SRC"
  tar xzf "$TARBALL" -C "$SRC" --strip-components=1
fi
info "source    $SRC"

# ---- 3. assemble a toolkit root ---------------------------------------------
# CUDA_WHEEL_PYTHON=/path/to/python points at the interpreter whose site-packages
# holds the nvidia/* wheels, if it is somewhere unusual.
# CMake's FindCUDAToolkit expects bin/ include/ lib64/ in one tree. A conda
# prefix already looks like that; pip wheels scatter the parts, so we stitch
# symlinks into a tree that satisfies it.
say "3/5  Preparing the toolkit layout"
TK="$(cd "$(dirname "$NVCC")/.." && pwd)"
if [ -e "$TK/include/cuda_runtime.h" ] && ls "$TK"/lib*/libcudart.so* >/dev/null 2>&1; then
  info "complete toolkit at $TK"
else
  STITCH="$PREFIX/cuda-toolkit"
  info "incomplete toolkit — stitching one at $STITCH"
  rm -rf "$STITCH"; mkdir -p "$STITCH/bin" "$STITCH/include" "$STITCH/lib64"
  cp -a "$(dirname "$NVCC")/." "$STITCH/bin/" 2>/dev/null || true
  [ -d "$TK/nvvm" ] && cp -a "$TK/nvvm" "$STITCH/" 2>/dev/null || true
  # nvcc also needs its own targets/ tree, where the conda package keeps headers
  for d in "$TK/targets/x86_64-linux/include" "$TK/targets/x86_64-linux/lib"; do
    [ -d "$d" ] || continue
    case "$d" in *include) cp -asn "$d/." "$STITCH/include/" 2>/dev/null || true ;;
                  *lib)     cp -asn "$d/." "$STITCH/lib64/"   2>/dev/null || true ;; esac
  done
  # cudart and cuBLAS usually come from PyTorch's pip wheels. Ask several pythons
  # where those live: PATH here may point at a build-only interpreter (the conda
  # env we just made) whose site-packages has no nvidia/ at all.
  PYSPS=()
  for py in ${CUDA_WHEEL_PYTHON:-} /usr/bin/python3 python3 \
            "$HOME"/anaconda3/bin/python3 "$HOME"/miniconda3/bin/python3 \
            "$HOME"/.local/bin/python3 /usr/local/bin/python3; do
    [ -n "$py" ] || continue
    command -v "$py" >/dev/null 2>&1 || continue
    sp="$("$py" -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null)" || continue
    [ -d "$sp/nvidia" ] || continue
    case " ${PYSPS[*]:-} " in *" $sp "*) ;; *) PYSPS+=("$sp") ;; esac
  done
  [ ${#PYSPS[@]} -gt 0 ] && info "cuda libs from ${PYSPS[0]}/nvidia" || true

  for d in "$TK/include" "$TK/targets/x86_64-linux/include"; do
    [ -d "$d" ] && cp -asn "$d/." "$STITCH/include/" 2>/dev/null || true
  done
  for d in "$TK/lib64" "$TK/lib" "$TK/targets/x86_64-linux/lib"; do
    [ -d "$d" ] && cp -asn "$d/." "$STITCH/lib64/" 2>/dev/null || true
  done
  for sp in "${PYSPS[@]:-}"; do
    [ -n "$sp" ] || continue
    for d in "$sp"/nvidia/*/include; do
      [ -d "$d" ] && cp -asn "$d/." "$STITCH/include/" 2>/dev/null || true
    done
    for d in "$sp"/nvidia/*/lib; do
      [ -d "$d" ] && cp -asn "$d/." "$STITCH/lib64/" 2>/dev/null || true
    done
  done
  # Linking wants the unversioned names; the system only ships versioned ones.
  # Note the trailing `|| true` on each: under `set -e`, a bare `[ ... ] && cmd`
  # that tests false is a failing last command and would abort the script with no
  # message at all.
  for base in libcudart libcublas libcublasLt; do
    f="$(ls "$STITCH"/lib64/$base.so.* 2>/dev/null | head -1)" || true
    [ -n "$f" ] && ln -sf "$(basename "$f")" "$STITCH/lib64/$base.so" || true
  done
  for drv in /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib64/libcuda.so.1; do
    [ -e "$drv" ] && { ln -sf "$drv" "$STITCH/lib64/libcuda.so"; break; } || true
  done
  [ -e "$STITCH/include/cuda_runtime.h" ] || die "still no cuda_runtime.h — install cuda-cudart-dev"
  [ -d "$STITCH/include/cub" ] || die "no CUB headers — install cuda-cccl"
  ls "$STITCH"/lib64/libcublas.so* >/dev/null 2>&1 \
    || die "no cuBLAS library found. It usually arrives with PyTorch (pip install nvidia-cublas-cu12), or via conda's libcublas-dev"
  TK="$STITCH"
fi
info "toolkit   $TK"

# ---- 4. build ----------------------------------------------------------------
say "4/5  Building (this takes a while; $JOBS jobs)"
BUILD_DIR="$SRC/build-cuda"
rm -rf "$BUILD_DIR/CMakeCache.txt" "$BUILD_DIR/CMakeFiles"   # stale cache keeps old paths
cmake -S "$SRC" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCUDAToolkit_ROOT="$TK" \
  -DCUDAToolkit_INCLUDE_DIR="$TK/include" \
  -DCMAKE_LIBRARY_PATH="$TK/lib64" \
  -DCMAKE_INCLUDE_PATH="$TK/include" \
  -DCMAKE_CUDA_FLAGS="-I$TK/include" \
  -DCMAKE_C_FLAGS="-I$TK/include" \
  -DCMAKE_CXX_FLAGS="-I$TK/include" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_CURL=OFF \
  >"$PREFIX/src/cmake-configure.log" 2>&1 \
  || { tail -30 "$PREFIX/src/cmake-configure.log"; die "cmake configure failed (full log: $PREFIX/src/cmake-configure.log)"; }
info "configured"

cmake --build "$BUILD_DIR" -j "$JOBS" >"$PREFIX/src/cmake-build.log" 2>&1 \
  || { tail -40 "$PREFIX/src/cmake-build.log"; die "build failed (full log: $PREFIX/src/cmake-build.log)"; }
info "built"

# ---- 5. install alongside ----------------------------------------------------
say "5/5  Installing alongside the existing build"
DEST="$PREFIX/${SRC_BUILD}-cuda"
rm -rf "$DEST"; mkdir -p "$DEST"
find "$BUILD_DIR/bin" -maxdepth 1 -type f -exec cp -a {} "$DEST/" \; 2>/dev/null || true
find "$BUILD_DIR" -name "*.so*" -exec cp -a {} "$DEST/" \; 2>/dev/null || true
[ -x "$DEST/llama-server" ] || die "llama-server missing from the build output"
info "installed $DEST"

cat <<NEXT

  Built, but not switched to. Measure first:

    LLAMA_DIR=$DEST local-claude bench --tune
    local-claude bench --tune                    # the current build, for comparison

  If CUDA wins, adopt it:

    ln -sfn $DEST $PREFIX/current
    local-claude stop && local-claude bench --tune

  The Vulkan build stays in place either way; repoint the symlink to go back.
NEXT
