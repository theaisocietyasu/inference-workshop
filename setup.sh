#!/usr/bin/env bash
# Installs llama.cpp and downloads the workshop model. Safe to run again: it skips
# whatever is already in place. Non-interactive, so it also works as a pod setup script.
#
# Environment (all optional):
#   LLAMA_HOME     where binaries and models go (default: <repo>/.llama)
#   LLAMA_TAG      llama.cpp release tag (default below)
#   BACKEND        auto | cpu | cuda | metal (default: auto)
#   MODEL_URL      GGUF file to download (default: Qwen2.5 0.5B Instruct, 4-bit)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_HOME="${LLAMA_HOME:-$REPO_DIR/.llama}"
LLAMA_TAG="${LLAMA_TAG:-b12119}"
BACKEND="${BACKEND:-auto}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
RELEASES="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG"

log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing '$1'. Install it and run again."; }
need curl
need tar

mkdir -p "$LLAMA_HOME/models"

# Downloads a release tarball and unpacks it into $2.
fetch_release() {
  local asset="$1" dest="$2" tmp
  tmp="$(mktemp -d)"
  log "downloading $asset"
  curl -fL --retry 3 --progress-bar -o "$tmp/a.tar.gz" "$RELEASES/$asset" || { rm -rf "$tmp"; return 1; }
  mkdir -p "$dest"
  tar -xzf "$tmp/a.tar.gz" -C "$dest" --strip-components=1
  rm -rf "$tmp"
}

# Prints the glibc version as an integer, e.g. 2.35 -> 235.
glibc_num() {
  local v
  v="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0.0)"
  echo "${v%%.*}$(printf '%02d' "${v##*.}")"
}

has_nvidia() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

works() { [ -x "$1/llama-server" ] && "$1/llama-server" --version >/dev/null 2>&1; }

# Builds llama-server with CUDA from source. Used when the prebuilt CUDA binaries
# need a newer glibc than the machine has (Ubuntu 22.04 images, for example).
build_cuda() {
  local src="$LLAMA_HOME/src-$LLAMA_TAG" dest="$1" arch
  local nvcc
  nvcc="$(command -v nvcc || true)"
  [ -z "$nvcc" ] && [ -x /usr/local/cuda/bin/nvcc ] && nvcc=/usr/local/cuda/bin/nvcc
  [ -n "$nvcc" ] || return 1
  need git
  command -v c++ >/dev/null 2>&1 || return 1
  if ! command -v cmake >/dev/null 2>&1; then
    log "installing cmake for this user with pip"
    python3 -m pip install --user --quiet cmake || return 1
    export PATH="$HOME/.local/bin:$PATH"
  fi
  arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
  [ -d "$src/.git" ] || git clone --quiet --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$src"
  log "building llama.cpp $LLAMA_TAG with CUDA for sm_$arch (takes 5 to 15 minutes, once)"
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="$nvcc" -DCMAKE_CUDA_ARCHITECTURES="$arch" \
    -DLLAMA_BUILD_NUMBER="${LLAMA_TAG#b}" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF >/dev/null
  cmake --build "$src/build" --config Release -j "$(nproc)" --target llama-server llama-bench
  mkdir -p "$dest"
  cp -a "$src/build/bin/." "$dest/"
}

install_cpu() {
  local os="$1" arch="$2" dest="$LLAMA_HOME/$LLAMA_TAG-cpu"
  works "$dest" && { echo "$dest"; return; }
  fetch_release "llama-$LLAMA_TAG-bin-$os-$arch.tar.gz" "$dest" >&2 || die "download failed"
  echo "$dest"
}

install_cuda() {
  local arch="$1" dest="$LLAMA_HOME/$LLAMA_TAG-cuda" label driver
  works "$dest" && { echo "$dest"; return 0; }
  rm -rf "$dest"
  if [ "$(glibc_num)" -ge 238 ]; then
    driver="$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+' | grep -oE '[0-9]+$' || echo 12)"
    label="12.8"
    { [ "$driver" -ge 13 ] || [ "$arch" = arm64 ]; } && label="13.4"
    fetch_release "llama-$LLAMA_TAG-bin-ubuntu-cuda-$label-$arch.tar.gz" "$dest" >&2 &&
      fetch_release "cudart-llama-$LLAMA_TAG-bin-ubuntu-cuda-$label-$arch.tar.gz" "$dest" >&2 &&
      works "$dest" && { echo "$dest"; return 0; }
    warn "prebuilt CUDA binaries did not start"
  fi
  [ "$arch" = x64 ] && build_cuda "$dest" >&2 && works "$dest" && { echo "$dest"; return 0; }
  rm -rf "$dest"
  return 1
}

# Pick and install a backend.
os="$(uname -s)"
case "$(uname -m)" in
  x86_64 | amd64) arch=x64 ;;
  aarch64 | arm64) arch=arm64 ;;
  *) die "unsupported CPU: $(uname -m)" ;;
esac

if [ "$os" = Darwin ]; then
  BACKEND=metal
  dir="$LLAMA_HOME/$LLAMA_TAG-metal"
  works "$dir" || fetch_release "llama-$LLAMA_TAG-bin-macos-$arch.tar.gz" "$dir" || die "download failed"
  xattr -dr com.apple.quarantine "$dir" 2>/dev/null || true
elif [ "$os" = Linux ]; then
  if [ "$BACKEND" = auto ]; then has_nvidia && BACKEND=cuda || BACKEND=cpu; fi
  if [ "$BACKEND" = cuda ]; then
    if ! dir="$(install_cuda "$arch")"; then
      warn "no CUDA build possible here (needs glibc 2.38+, or nvcc to build). Falling back to CPU."
      BACKEND=cpu
    fi
  fi
  [ "$BACKEND" = cpu ] && dir="$(install_cpu ubuntu "$arch")"
else
  die "unsupported OS: $os. On Windows, use WSL."
fi
works "$dir" || die "llama-server in $dir does not start"

# Download the model.
model="$LLAMA_HOME/models/$(basename "${MODEL_URL%%\?*}")"
if [ ! -s "$model" ]; then
  log "downloading model $(basename "$model")"
  curl -fL --retry 3 --progress-bar -o "$model.part" "$MODEL_URL" || die "model download failed"
  mv "$model.part" "$model"
fi

# run.sh reads this file.
cat >"$LLAMA_HOME/env" <<EOF
LLAMA_BIN="$dir"
BACKEND="$BACKEND"
MODEL="$model"
EOF
log "ready: backend=$BACKEND, model=$(basename "$model")"
log "next: ./run.sh"
