#!/bin/zsh
set -euo pipefail
mkdir -p outputs/test_colored_batch_v2
: > outputs/test_colored_batch_v2/metrics.ndjson
find gt_rgb/test -type f -name '*.png' | sort | while read -r path; do
  echo "$path"
done | xargs -I{} -P 8 zsh -c 'python3 batch_eval.py "$1" >> outputs/test_colored_batch_v2/metrics.ndjson' _ {}
