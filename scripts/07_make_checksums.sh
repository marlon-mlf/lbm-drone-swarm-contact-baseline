#!/usr/bin/env bash
set -euo pipefail

# Rebuild checksum and manifest files for archive contents.
# MANIFEST.tsv and checksums_sha256.txt are excluded from their own hashes to
# avoid self-referential checksum changes.

OUT=${OUT:-checksums_sha256.txt}
MANIFEST=${MANIFEST:-MANIFEST.tsv}

find . -type f \
  ! -path "./$OUT" \
  ! -path "./$MANIFEST" \
  ! -path "./.venv/*" \
  -printf '%P\0' | sort -z | xargs -0 sha256sum > "$OUT"

{
  printf "path\tsize_bytes\tsha256\n"
  while IFS= read -r -d '' f; do
    size=$(stat -c '%s' "$f")
    sha=$(sha256sum "$f" | awk '{print $1}')
    printf "%s\t%s\t%s\n" "${f#./}" "$size" "$sha"
  done < <(find . -type f \
    ! -path "./$OUT" \
    ! -path "./$MANIFEST" \
    ! -path "./.venv/*" \
    -print0 | sort -z)
} > "$MANIFEST"

echo "Wrote $OUT and $MANIFEST"
