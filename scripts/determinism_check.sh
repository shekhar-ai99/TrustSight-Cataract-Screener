#!/usr/bin/env bash
# Simple determinism check: run inference twice and compare SHA256 of outputs
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <image> [weights]"
  exit 2
fi
IMAGE=$1
WEIGHTS=${2:-}
OUT1=/tmp/infer_out1.json
OUT2=/tmp/infer_out2.json
if [ -n "$WEIGHTS" ]; then
  python3 - <<PY > "$OUT1"
from src.inference import infer
print(infer("$IMAGE", explain=False, n_mc=15, weights_path="$WEIGHTS"))
PY
  python3 - <<PY > "$OUT2"
from src.inference import infer
print(infer("$IMAGE", explain=False, n_mc=15, weights_path="$WEIGHTS"))
PY
else
  python3 - <<PY > "$OUT1"
from src.inference import infer
print(infer("$IMAGE", explain=False, n_mc=15))
PY
  python3 - <<PY > "$OUT2"
from src.inference import infer
print(infer("$IMAGE", explain=False, n_mc=15))
PY
fi
sha1=$(sha256sum "$OUT1" | awk '{print $1}')
sha2=$(sha256sum "$OUT2" | awk '{print $1}')
echo "sha1: $sha1"
echo "sha2: $sha2"
if [ "$sha1" = "$sha2" ]; then
  echo "DETERMINISTIC: outputs identical"
  exit 0
else
  echo "NON-DETERMINISTIC: outputs differ"
  exit 1
fi
