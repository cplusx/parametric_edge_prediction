#!/bin/zsh
set -euo pipefail
: > outputs/test_colored_batch/metrics.ndjson
find gt_rgb/test -type f -name '*.png' | sort | while read -r path; do
  echo "$path"
done | xargs -I{} -P 8 zsh -c 'python3 batch_eval.py "$1" >> outputs/test_colored_batch/metrics.ndjson' _ {}
