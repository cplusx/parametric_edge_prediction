#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/viplab/anaconda3/envs/diffusers/bin/python}"
BASE_CONFIG="configs/parametric_edge/archived_overfit/overfit_diverse16_2000_memorization_base.yaml"
SUMMARY_SCRIPT="scripts/summarize_addback_ablation.py"

run_experiment() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] Starting ${name}"
  (cd "$ROOT_DIR" && "$PYTHON_BIN" train.py "$@")
  echo "[$(date '+%F %T')] Finished ${name}"
  (cd "$ROOT_DIR" && "$PYTHON_BIN" "$SUMMARY_SCRIPT")
}

run_experiment memorization --config "$BASE_CONFIG"
run_experiment aux --config "$BASE_CONFIG" --override-config configs/parametric_edge/archived_overfit/overfit_diverse16_2000_addback_aux.yaml
run_experiment dn --config "$BASE_CONFIG" --override-config configs/parametric_edge/archived_overfit/overfit_diverse16_2000_addback_dn.yaml
run_experiment one_to_many --config "$BASE_CONFIG" --override-config configs/parametric_edge/archived_overfit/overfit_diverse16_2000_addback_onetomany.yaml
run_experiment topk --config "$BASE_CONFIG" --override-config configs/parametric_edge/archived_overfit/overfit_diverse16_2000_addback_topk.yaml
run_experiment distinct --config "$BASE_CONFIG" --override-config configs/parametric_edge/archived_overfit/overfit_diverse16_2000_addback_distinct.yaml

echo "[$(date '+%F %T')] Full overfit add-back suite completed"