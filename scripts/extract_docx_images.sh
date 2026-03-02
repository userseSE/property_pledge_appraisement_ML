#!/usr/bin/env bash
set -euo pipefail

# Extract images from a .docx file (Word is a zip container)
# Usage:
#   bash scripts/extract_docx_images.sh "doc/your_file.docx" "assets/figures/docx_extracted"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/extract_docx_images.sh <docx_path> [output_dir]"
  exit 1
fi

DOCX_PATH="$1"
OUT_DIR="${2:-assets/figures/docx_extracted}"

if [[ ! -f "$DOCX_PATH" ]]; then
  echo "File not found: $DOCX_PATH"
  exit 1
fi

mkdir -p "$OUT_DIR"

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

unzip -o "$DOCX_PATH" "word/media/*" -d "$tmpdir" >/dev/null

if [[ ! -d "$tmpdir/word/media" ]]; then
  echo "No media files found in $DOCX_PATH"
  exit 0
fi

cp -f "$tmpdir"/word/media/* "$OUT_DIR"/
echo "Extracted $(find "$tmpdir/word/media" -type f | wc -l | tr -d ' ') files to $OUT_DIR"

