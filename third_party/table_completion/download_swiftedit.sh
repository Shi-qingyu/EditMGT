#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../SwiftEdit" && pwd)"
if [[ -d "$root/swiftedit_weights/inverse_ckpt-120k" ]]; then
    echo "SwiftEdit weights already ready in $root/swiftedit_weights"
    exit 0
fi
mkdir -p "$root/weights_download"
cd "$root/weights_download"

base='https://github.com/Qualcomm-AI-research/SwiftEdit/releases/download/v1.0'
for suffix in aa ab ac ad ae; do
    part="swiftedit_weights.tar.gz.part-$suffix"
    curl -fL --retry 8 --retry-delay 5 -C - -o "$part" "$base/$part"
done

cat swiftedit_weights.tar.gz.part-* > swiftedit_weights.tar.gz
tar -xzf swiftedit_weights.tar.gz -C "$root"
test -d "$root/swiftedit_weights/inverse_ckpt-120k"
echo "SwiftEdit weights ready in $root/swiftedit_weights"
