#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/jiaxin/parametric_edge_prediction"
DATA_ROOT="/data/jiaxin/laion/edge_detection"
CONFIG_PATH="configs/parametric_edge/laion_curve_pretrain_lab34_v3_edgeprob05.yaml"

set +u
source /home/viplab/anaconda3/etc/profile.d/conda.sh
conda activate diffusers
set -u

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/rewrite_bezier_entry_cache.py \
  --input-cache "${DATA_ROOT}/laion_entry_cache_v3_bezier.txt" \
  --output-cache "${DATA_ROOT}/laion_entry_cache_v3_bezier.txt" \
  --image-root "${DATA_ROOT}" \
  --bezier-root "${DATA_ROOT}/laion_edge_v3_bezier"

python train.py --config "${CONFIG_PATH}"
