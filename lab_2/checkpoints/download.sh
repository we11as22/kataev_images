#!/usr/bin/env bash
# Download SAM 2.1 tiny checkpoint (~149 MB) into this directory.
set -euo pipefail

cd "$(dirname "$0")"
CKPT="sam2.1_hiera_tiny.pt"
URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"

if [ -f "$CKPT" ]; then
  echo "Already present: $CKPT ($(du -h "$CKPT" | cut -f1))"
  exit 0
fi

echo "Downloading $CKPT ..."
if command -v wget >/dev/null 2>&1; then
  wget -O "$CKPT" "$URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L -o "$CKPT" "$URL"
else
  echo "Install wget or curl, or download manually:" >&2
  echo "  $URL" >&2
  exit 1
fi

echo "Saved: $(pwd)/$CKPT"
