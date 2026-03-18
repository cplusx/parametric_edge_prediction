#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GPUS=4
PARTITION=""
NODELIST=""
TIME_LIMIT=""
CPUS=0
MEMORY=""
PER_DEVICE_BATCH=32
NUM_WORKERS=16
NUM_QUERIES=640
EFFECTIVE_BATCH=256
EFFECTIVE_STEPS_PER_EPOCH=2000
MAX_EPOCHS=200
VAL_SAMPLES=1024
TEST_SAMPLES=1024
SELECTION_SEED=20260318
TRAIN_SELECTION_OFFSET=2048
PRECISION="16-mixed"
RUN_NAME="laion-pretrain-q640-eb256-lr5e5"
CONDA_ENV="diffusers"
OUTPUT_ROOT="${CLUSTER_OUTPUT_ROOT:-$HOME/cluster_runs/parametric_edge_prediction}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --nodelist)
      NODELIST="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --mem)
      MEMORY="$2"
      shift 2
      ;;
    --per-device-batch)
      PER_DEVICE_BATCH="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --num-queries)
      NUM_QUERIES="$2"
      shift 2
      ;;
    --effective-batch)
      EFFECTIVE_BATCH="$2"
      shift 2
      ;;
    --effective-steps-per-epoch)
      EFFECTIVE_STEPS_PER_EPOCH="$2"
      shift 2
      ;;
    --max-epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PARTITION" ]]; then
  echo "--partition is required" >&2
  exit 1
fi

if [[ -z "$TIME_LIMIT" ]]; then
  echo "--time is required" >&2
  exit 1
fi

if (( CPUS <= 0 )); then
  CPUS=$(( GPUS * 16 ))
fi

if [[ -z "$MEMORY" ]]; then
  MEMORY="$(( GPUS * 64 ))G"
fi

SUBMIT_TOKEN="${RUN_NAME}-$(date +%Y%m%d-%H%M%S)-${GPUS}gpu"
OUTPUT_ROOT="${OUTPUT_ROOT/#\~/$HOME}"
SUBMIT_DIR="$OUTPUT_ROOT/${SUBMIT_TOKEN}"
mkdir -p "$SUBMIT_DIR"
SBATCH_FILE="$SUBMIT_DIR/launch.sbatch"
STDOUT_FILE="$SUBMIT_DIR/slurm-%j.out"
STDERR_FILE="$SUBMIT_DIR/slurm-%j.err"
CONFIG_FILE="$SUBMIT_DIR/resolved_cluster_config.yaml"

if (( GPUS <= 0 )); then
  echo "--gpus must be positive" >&2
  exit 1
fi

GLOBAL_MICRO_BATCH=$(( GPUS * PER_DEVICE_BATCH ))
if (( EFFECTIVE_BATCH % GLOBAL_MICRO_BATCH != 0 )); then
  echo "effective batch ${EFFECTIVE_BATCH} is not divisible by gpus*per_device_batch (${GPUS}*${PER_DEVICE_BATCH}=${GLOBAL_MICRO_BATCH})" >&2
  exit 1
fi

ACCUMULATE=$(( EFFECTIVE_BATCH / GLOBAL_MICRO_BATCH ))
RUN_TOKEN="$SUBMIT_TOKEN"
LAION_ROOT="${LAION_ROOT:-$HOME/laion/edge_detection}"
LAION_IMAGE_ROOT="${LAION_IMAGE_ROOT:-}"
if [[ -z "$LAION_IMAGE_ROOT" ]]; then
  if [[ -d "$LAION_ROOT/edge_detection" ]]; then
    LAION_IMAGE_ROOT="$LAION_ROOT/edge_detection"
  else
    LAION_IMAGE_ROOT="$LAION_ROOT"
  fi
fi
LAION_CACHE_ROOT="$LAION_ROOT/laion_edge_v2_bezier_cache_fast"
LAION_EDGE_ROOT="$LAION_ROOT/laion_edge_v2"

SBATCH_ARGS=()
if [[ -n "$NODELIST" ]]; then
  SBATCH_ARGS+=("#SBATCH -w ${NODELIST}")
fi

cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH -J ${RUN_NAME}
#SBATCH -p ${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH -c ${CPUS}
#SBATCH --mem=${MEMORY}
#SBATCH -t ${TIME_LIMIT}
#SBATCH -o ${STDOUT_FILE}
#SBATCH -e ${STDERR_FILE}
${SBATCH_ARGS[*]}

set -euo pipefail
set +u
source "\$HOME/anaconda3/etc/profile.d/conda.sh"
if command -v conda >/dev/null 2>&1; then
  conda activate ${CONDA_ENV}
else
  source "\$HOME/anaconda3/bin/activate" ${CONDA_ENV}
fi
set -u
export PYTHONUNBUFFERED=1
cd ${REPO_ROOT}

cat > "${CONFIG_FILE}" <<CFGEOF
data:
  include_primary_train_dataset: false
  batch_size: ${PER_DEVICE_BATCH}
  val_batch_size: ${PER_DEVICE_BATCH}
  num_workers: ${NUM_WORKERS}
  extra_train_datasets:
    - dataset_type: laion_synthetic
      data_root: ${LAION_ROOT}
      image_root: ${LAION_IMAGE_ROOT}
      edge_root: ${LAION_EDGE_ROOT}
      cache_root: ${LAION_CACHE_ROOT}
      batch_glob: batch*
      quantize: 4
      selection_seed: ${SELECTION_SEED}
      selection_offset: ${TRAIN_SELECTION_OFFSET}
  val_dataset:
    dataset_type: laion_synthetic
    data_root: ${LAION_ROOT}
    image_root: ${LAION_IMAGE_ROOT}
    edge_root: ${LAION_EDGE_ROOT}
    cache_root: ${LAION_CACHE_ROOT}
    batch_glob: batch*
    quantize: 4
    selection_seed: ${SELECTION_SEED}
    selection_offset: 0
    max_samples: ${VAL_SAMPLES}
  test_dataset:
    dataset_type: laion_synthetic
    data_root: ${LAION_ROOT}
    image_root: ${LAION_IMAGE_ROOT}
    edge_root: ${LAION_EDGE_ROOT}
    cache_root: ${LAION_CACHE_ROOT}
    batch_glob: batch*
    quantize: 4
    selection_seed: ${SELECTION_SEED}
    selection_offset: ${VAL_SAMPLES}
    max_samples: ${TEST_SAMPLES}

model:
  num_queries: ${NUM_QUERIES}

optimizer:
  lr: 0.00005
  backbone_lr: 0.000025

trainer:
  default_root_dir: ${SUBMIT_DIR}
  devices: ${GPUS}
  precision: ${PRECISION}
  accumulate_grad_batches: ${ACCUMULATE}
  max_epochs: ${MAX_EPOCHS}
  effective_train_batches_per_epoch: ${EFFECTIVE_STEPS_PER_EPOCH}
  check_val_every_n_epoch: 1
  num_sanity_val_steps: 0
  run_test_after_fit: false

callbacks:
  visualization_every_n_epochs: 1
  visualization_every_n_train_steps: ${EFFECTIVE_STEPS_PER_EPOCH}

logging:
  wandb:
    name: ${RUN_TOKEN}
    group: laion-cluster-pretrain
    job_type: train
    save_dir: ${SUBMIT_DIR}/wandb
    tags:
      - laion
      - cluster
      - ${GPUS}gpu
      - effective-batch-${EFFECTIVE_BATCH}
CFGEOF

echo "Resolved config written to: ${CONFIG_FILE}"
echo "GPUs=${GPUS} per_device_batch=${PER_DEVICE_BATCH} accumulate=${ACCUMULATE} effective_batch=${EFFECTIVE_BATCH}"
echo "LAION_ROOT=${LAION_ROOT}"
echo "LAION_IMAGE_ROOT=${LAION_IMAGE_ROOT}"
echo "OUTPUT_DIR=${SUBMIT_DIR}"

exec python train.py --config configs/parametric_edge/default.yaml --override-config "${CONFIG_FILE}"
EOF

chmod +x "$SBATCH_FILE"
echo "Created sbatch script: $SBATCH_FILE"
echo "Stdout: $STDOUT_FILE"
echo "Stderr: $STDERR_FILE"

if (( DRY_RUN == 1 )); then
  sed -n '1,220p' "$SBATCH_FILE"
  exit 0
fi

cd "$REPO_ROOT"
sbatch "$SBATCH_FILE"