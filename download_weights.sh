#!/usr/bin/env bash
# Download Hunyuan3D-2 model weights from HuggingFace
# Source: https://huggingface.co/tencent/Hunyuan3D-2
#
# Total size breakdown:
#   hunyuan3d-dit-v2-0          24.6 GB  (full DiT model)
#   hunyuan3d-dit-v2-0-fast      9.9 GB  (fast DiT variant)
#   hunyuan3d-dit-v2-0-turbo     9.9 GB  (turbo DiT variant)
#   hunyuan3d-vae-v2-0           0.9 GB  (VAE decoder)
#   hunyuan3d-vae-v2-0-turbo     0.8 GB  (VAE turbo)
#   hunyuan3d-vae-v2-0-withenc   1.3 GB  (VAE with encoder)
#   hunyuan3d-delight-v2-0       ~50 MB  (delighting model)
#   hunyuan3d-paint-v2-0         ~50 MB  (texture paint)
#   hunyuan3d-paint-v2-0-turbo   ~50 MB  (texture paint turbo)
#   ─────────────────────────────────────
#   TOTAL (all)                 ~47 GB
#   TOTAL (turbo only)          ~11 GB   ← recommended for M3 Mac
#
# Usage:
#   ./download_weights.sh              # Download turbo subset (~11 GB)
#   ./download_weights.sh --all        # Download everything (~47 GB)
#   ./download_weights.sh --dry-run    # Show what would be downloaded
#
# Requirements: git-lfs (brew install git-lfs)

set -euo pipefail

REPO="tencent/Hunyuan3D-2"
HF_BASE="https://huggingface.co/${REPO}/resolve/main"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Turbo subset — recommended for M3 Mac (smaller, still good quality)
TURBO_DIRS=(
  "hunyuan3d-dit-v2-0-turbo"
  "hunyuan3d-vae-v2-0-turbo"
  "hunyuan3d-delight-v2-0"
  "hunyuan3d-paint-v2-0-turbo"
)

# Additional dirs for --all
FULL_EXTRA_DIRS=(
  "hunyuan3d-dit-v2-0"
  "hunyuan3d-dit-v2-0-fast"
  "hunyuan3d-vae-v2-0"
  "hunyuan3d-vae-v2-0-withencoder"
  "hunyuan3d-paint-v2-0"
)

MODE="turbo"
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --all) MODE="all" ;;
    --dry-run) DRY_RUN=true ;;
    --help|-h)
      head -25 "$0" | grep "^#" | sed 's/^# \?//'
      exit 0
      ;;
  esac
done

if [ "$MODE" = "all" ]; then
  DIRS=("${TURBO_DIRS[@]}" "${FULL_EXTRA_DIRS[@]}")
  echo "📦 Downloading ALL weights (~47 GB)..."
else
  DIRS=("${TURBO_DIRS[@]}")
  echo "📦 Downloading TURBO weights (~11 GB)..."
fi

if $DRY_RUN; then
  echo "🔍 DRY RUN — nothing will be downloaded."
  echo ""
fi

download_dir() {
  local dir="$1"
  local target="${SCRIPT_DIR}/${dir}"
  
  echo ""
  echo "━━━ ${dir} ━━━"
  
  # Get file list from HF API
  local files
  files=$(python3 -c "
import json, urllib.request
url = 'https://huggingface.co/api/models/${REPO}/tree/main/${dir}'
data = json.loads(urllib.request.urlopen(url).read())
for f in data:
    if f['type'] == 'file':
        print(f'{f[\"size\"]}\t{f[\"path\"]}')
" 2>/dev/null)

  if [ -z "$files" ]; then
    echo "  ⚠️  No files found (or API error)"
    return
  fi

  mkdir -p "$target"

  while IFS=$'\t' read -r size path; do
    local filename
    filename=$(basename "$path")
    local size_mb
    size_mb=$(echo "scale=1; $size / 1000000" | bc 2>/dev/null || echo "?")

    if [ -f "${target}/${filename}" ]; then
      echo "  ✅ ${filename} (${size_mb} MB) — already exists, skipping"
      continue
    fi

    if $DRY_RUN; then
      echo "  📥 Would download: ${filename} (${size_mb} MB)"
    else
      echo "  📥 Downloading: ${filename} (${size_mb} MB)..."
      curl -L --progress-bar -o "${target}/${filename}" \
        "${HF_BASE}/${path}"
    fi
  done <<< "$files"
}

# Also grab config.json from root
echo ""
if $DRY_RUN; then
  echo "📥 Would download: config.json"
else
  echo "📥 Downloading: config.json..."
  curl -sL -o "${SCRIPT_DIR}/config.json" "${HF_BASE}/config.json" 2>/dev/null || true
fi

for dir in "${DIRS[@]}"; do
  download_dir "$dir"
done

echo ""
if $DRY_RUN; then
  echo "✅ Dry run complete. Re-run without --dry-run to download."
else
  echo "✅ All weights downloaded to: ${SCRIPT_DIR}"
  echo ""
  echo "To run on M3 Mac, you'll need:"
  echo "  pip install -r requirements.txt"
  echo "  python minimal_demo.py --model_path . --use_turbo"
fi
