#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/viplab/anaconda3/envs/diffusers/bin/python}"
BASE_CONFIG="configs/parametric_edge/formal_bsds_train_val_test.yaml"

CONFIGS=(
  "configs/parametric_edge/formal_bsds_loss_search_balanced_soft_50ep.yaml"
  "configs/parametric_edge/formal_bsds_loss_search_balanced_base_50ep.yaml"
  "configs/parametric_edge/formal_bsds_loss_search_balanced_reg_50ep.yaml"
)

for override_config in "${CONFIGS[@]}"; do
  echo "[$(date '+%F %T')] Starting $override_config"
  CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" train.py --config "$BASE_CONFIG" --override-config "$override_config"
  echo "[$(date '+%F %T')] Finished $override_config"
done
