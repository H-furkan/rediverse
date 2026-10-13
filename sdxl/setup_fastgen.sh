#!/usr/bin/env bash
# Prepare FastGen for SDXL ReDiverse-FT training: clone NVlabs/FastGen (Apache-2.0) at the pinned
# commit into third_party/FastGen and add the ReDiverse files from sdxl/fastgen_overlay. No upstream
# file is modified; the added files are listed in REDIVERSE_OVERLAY.md inside the checkout.
#
#   bash sdxl/setup_fastgen.sh                    # -> third_party/FastGen
#   bash sdxl/setup_fastgen.sh /path/to/FastGen   # another location (pass it to sdxl.train --fastgen)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
DEST="${1:-$ROOT/third_party/FastGen}"
REPO=https://github.com/NVlabs/FastGen
COMMIT=060b8570a62c105c1a8a4f8b6f2e8089a3f19666
OVERLAY="$HERE/fastgen_overlay"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone "$REPO" "$DEST"
fi
git -C "$DEST" cat-file -e "$COMMIT^{commit}" 2>/dev/null || git -C "$DEST" fetch origin
git -C "$DEST" checkout -q "$COMMIT"

# the overlay only adds files: refuse to overwrite anything tracked upstream
cd "$OVERLAY"
while IFS= read -r f; do
  f="${f#./}"
  if git -C "$DEST" cat-file -e "$COMMIT:$f" 2>/dev/null; then
    echo "error: $f exists in FastGen $COMMIT; the overlay must not replace upstream files" >&2
    exit 1
  fi
done < <(find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*')
tar -C "$OVERLAY" -cf - --exclude=__pycache__ --exclude='*.pyc' . | tar -C "$DEST" -xf -

echo "FastGen $COMMIT ready at $DEST with the ReDiverse overlay:"
(cd "$OVERLAY" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | sed 's|^\./|  |' | sort)
