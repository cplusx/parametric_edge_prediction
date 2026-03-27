# Parametric Edge Experiment Log

This document tracks the active training entrypoints, run conventions, and current operational rules for this repository.

For historical ablations, see [docs/parametric_edge_ablation_log.md](./parametric_edge_ablation_log.md).

For lab-machine and cluster workflow rules, see [docs/lab_cluster_practical_manual.md](./lab_cluster_practical_manual.md).

## Environment

Typical working directories:

```bash
# lab30
cd /home/viplab/jiaxin/parametric_edge_prediction
conda activate diffusers

# cluster checkout
cd ~/parametric_edge_prediction
conda activate diffusers
```

Core entrypoint:

```bash
python train.py --config <config.yaml>
python train.py --config <base_config.yaml> --override-config <override.yaml>
```

Config loading behavior:

- `configs/parametric_edge/default.yaml` is merged first unless the chosen config sets `_inherit_default: false`.
- `--override-config` is merged on top of the already loaded config.
- Historical overfit configs live under `configs/parametric_edge/archived_overfit/`.

## Active Codebase Status

The active codebase now supports only the DAB mainline:

- model entry: `DABCurveDETR`
- model factory: [models/__init__.py](../models/__init__.py)
- implementation: [models/dab_curve_detr.py](../models/dab_curve_detr.py)

The old `ParametricDETR` entry is gone from the active code path.

The legacy grouped auxiliary stack has also been removed from the active codebase:

- one-to-many
- top-k grouped positives
- distinct grouped regularization

These are no longer part of the main code path, main configs, or current training flow. Historical references should be treated as ablation history only.

## Current Default Training

Primary configs:

- [configs/parametric_edge/default.yaml](../configs/parametric_edge/default.yaml)
- [configs/parametric_edge/laion_pretrain_cluster_dn_aux_ce.yaml](../configs/parametric_edge/laion_pretrain_cluster_dn_aux_ce.yaml)
- [configs/parametric_edge/laion_pretrain_cluster_dn_aux_ce_endpointheavy.yaml](../configs/parametric_edge/laion_pretrain_cluster_dn_aux_ce_endpointheavy.yaml)
- [configs/parametric_edge/bsds_formal.yaml](../configs/parametric_edge/bsds_formal.yaml)

Current mainline design:

- `DABCurveDETR`
- `ResNet50` backbone with ImageNet pretrained init
- `hidden_dim: 256`
- `nheads: 8`
- `6 encoder / 6 decoder`
- `num_queries: 300`
- global curve-coordinate remapping from external `[-0.1, 1.1]` to internal `[0, 1]`
- direction-invariant curve comparison
- single one-to-one Hungarian matching path
- focal classification in the main configs
- active geometry supervision limited to:
  - `ctrl`
  - `endpoint`

Current mainline does **not** use:

- grouped matching outputs
- one-to-many
- top-k grouped positives
- distinct
- `giou`
- `sample` loss
- `curve_distance` loss

DN support still exists inside `DABCurveDETR` as an optional training-time feature, but it is not part of the stripped mainline path unless explicitly enabled again in config.

LAION pretrain now uses a unified CE + DN + aux config, with an endpoint-heavy variant available:

- [configs/parametric_edge/laion_pretrain_cluster_dn_aux_ce.yaml](../configs/parametric_edge/laion_pretrain_cluster_dn_aux_ce.yaml)
- [configs/parametric_edge/laion_pretrain_cluster_dn_aux_ce_endpointheavy.yaml](../configs/parametric_edge/laion_pretrain_cluster_dn_aux_ce_endpointheavy.yaml)
- `trainer.devices` is resolved at runtime from the visible GPU count
- `trainer.accumulate_grad_batches` is inferred from `trainer.effective_batch_size`, current GPU count, and `data.batch_size`
- output dir and wandb name auto-format with `{devices}` so the same config can run on 2 GPU or 4 GPU nodes

## Current Optimizer / Runtime Defaults

LAION mainline:

- effective batch size `64`
- FP32
- warmup `10` epochs
- `lr: 2e-5`
- `backbone_lr: 1e-5`

BSDS formal:

- FP32
- warmup `10` epochs
- `lr: 2e-5`
- `backbone_lr: 1e-5`

## Output Conventions

Main training runs write to experiment-specific directories under:

- `outputs/parametric_edge_training/`

Cluster LAION configs write runtime outputs under:

- `~/cluster_runs/parametric_edge_prediction/`

Dataset-side graph caches are not tracked in Git.

## Cluster Submission Rules

Current repo discipline:

- edit committed config files first
- use direct `sbatch` for real cluster runs
- keep temporary batch scripts only on the cluster checkout
- do not commit cluster-only launcher scripts
- use explicit conda activation inside jobs

Current example:

```bash
sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH -J laion-dab-r50-q300-eb64-lr2e5-fp32-dn-aux-ce
#SBATCH -p gbunchQ
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 3-00:00:00
#SBATCH -o ~/cluster_runs/parametric_edge_prediction/%x-%j.out
#SBATCH -e ~/cluster_runs/parametric_edge_prediction/%x-%j.err

set -euo pipefail
set +u
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate diffusers
set -u
cd ~/parametric_edge_prediction
python train.py --config configs/parametric_edge/laion_pretrain_cluster_dn_aux_ce.yaml
EOF
```

## 2026-03-26 Cleanup Note

Repository cleanup completed around the DAB mainline:

- removed the old `ParametricDETR` entry from the active model factory
- removed the one-to-many / top-k / distinct legacy stack from the active code path
- trimmed the main configs to only the currently used DAB training path
- removed stale grouped-loss documentation from the main docs

If a document still describes grouped one-to-many training, treat it as stale unless it lives explicitly in the historical ablation log.
