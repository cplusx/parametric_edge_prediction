#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GPUS=2
PARTITION=""
NODELIST=""
TIME_LIMIT=""
CPUS=0
MEMORY=""
RUN_NAME="laion-pretrain-h100-2gpu-q640-eb256-lr5e5"
CONFIG_PATH="configs/parametric_edge/laion_pretrain_cluster.yaml"
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
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
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

if [[ ! -f "$REPO_ROOT/$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
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
RUNTIME_OVERRIDE_FILE="$SUBMIT_DIR/runtime_override.yaml"
RUN_TOKEN="$SUBMIT_TOKEN"

cp "$REPO_ROOT/$CONFIG_PATH" "$SUBMIT_DIR/base_config.yaml"

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

cat > "${RUNTIME_OVERRIDE_FILE}" <<CFGEOF
trainer:
  default_root_dir: ${SUBMIT_DIR}

logging:
  wandb:
    name: ${RUN_TOKEN}
    save_dir: ${SUBMIT_DIR}/wandb
CFGEOF

echo "Base config copied to: ${SUBMIT_DIR}/base_config.yaml"
echo "Runtime override written to: ${RUNTIME_OVERRIDE_FILE}"
echo "Output dir: ${SUBMIT_DIR}"

exec python train.py --config ${CONFIG_PATH} --override-config "${RUNTIME_OVERRIDE_FILE}"
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