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
RUN_NAME="laion-cluster-pretrain"
CONDA_ENV="diffusers"
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
SUBMIT_DIR="$REPO_ROOT/outputs/parametric_edge_training/${SUBMIT_TOKEN}"
mkdir -p "$SUBMIT_DIR"
SBATCH_FILE="$SUBMIT_DIR/launch.sbatch"
STDOUT_FILE="$SUBMIT_DIR/slurm-%j.out"
STDERR_FILE="$SUBMIT_DIR/slurm-%j.err"

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
conda activate ${CONDA_ENV}
set -u
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
cd ${REPO_ROOT}

./scripts/run_cluster_laion_pretrain_interactive.sh \
  --gpus ${GPUS} \
  --partition ${PARTITION} \
  --time ${TIME_LIMIT} \
  --per-device-batch ${PER_DEVICE_BATCH} \
  --num-workers ${NUM_WORKERS} \
  --num-queries ${NUM_QUERIES} \
  --effective-batch ${EFFECTIVE_BATCH} \
  --effective-steps-per-epoch ${EFFECTIVE_STEPS_PER_EPOCH} \
  --max-epochs ${MAX_EPOCHS} \
  --run-name ${RUN_NAME}
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