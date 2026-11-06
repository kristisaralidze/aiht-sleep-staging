#!/usr/bin/env bash
# Walch et al. 2019: "Motion and heart rate from a wrist-worn wearable and
# labeled sleep from polysomnography" (PhysioNet, sleep-accel v1.0.0).
# Downloads the release zip with wget (or curl if wget is missing), unpacks it
# into data/raw/ and verifies every file against PhysioNet's SHA256 list.
# Safe to re-run: exits early if data/raw/ already verifies, and resumes a
# partial download.
set -euo pipefail

URL="https://physionet.org/content/sleep-accel/get-zip/1.0.0/"
SUMS_URL="https://physionet.org/files/sleep-accel/1.0.0/SHA256SUMS.txt"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
ZIP="$RAW/sleep-accel-1.0.0.zip"
SUMS="$RAW/SHA256SUMS.txt"
STAGING="$RAW/.unzip_staging"

fetch() {
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$2" "$1"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -C - -o "$2" "$1"
  else
    echo "error: neither wget nor curl is installed" >&2
    exit 1
  fi
}

sha256_check() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum --quiet -c "$1"
  else
    shasum -a 256 --quiet -c "$1"
  fi
}

verify() {
  [ -f "$SUMS" ] || return 1
  (cd "$RAW" && sha256_check "$(basename "$SUMS")") >/dev/null 2>&1
}

mkdir -p "$RAW"

if verify; then
  echo "data/raw/ already complete and verified; nothing to do."
  exit 0
fi

# A complete zip makes curl -C - fail, so only fetch when the zip is not intact.
if [ -f "$ZIP" ] && unzip -tqq "$ZIP" >/dev/null 2>&1; then
  echo "Using existing $ZIP"
else
  echo "Downloading sleep-accel 1.0.0 (about 580 MB)"
  fetch "$URL" "$ZIP"
fi

echo "Unpacking"
rm -rf "$STAGING"
mkdir -p "$STAGING"
unzip -q "$ZIP" -d "$STAGING"

# The zip wraps everything in one top-level folder; lift its contents to data/raw/.
MOTION_DIR="$(find "$STAGING" -type d -name motion -print -quit)"
if [ -z "$MOTION_DIR" ]; then
  echo "error: no motion/ folder in $ZIP; unexpected zip layout" >&2
  exit 1
fi
SRC="$(dirname "$MOTION_DIR")"
for item in "$SRC"/*; do
  rm -rf "${RAW:?}/$(basename "$item")"
  mv "$item" "$RAW/"
done
rm -rf "$STAGING"

# Fetch beside the target so an interrupted download never looks like a real list.
if [ ! -f "$SUMS" ]; then
  fetch "$SUMS_URL" "$SUMS.part"
  mv "$SUMS.part" "$SUMS"
fi

if ! verify; then
  echo "error: checksum verification failed; keeping $ZIP for inspection" >&2
  (cd "$RAW" && sha256_check "$(basename "$SUMS")") >&2 || true
  exit 1
fi

rm -f "$ZIP"
echo "Downloaded and verified: data/raw/ (motion/, heart_rate/, labels/, steps/)"
