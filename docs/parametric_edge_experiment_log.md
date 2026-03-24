# Parametric Edge Experiment Log

This document tracks active training entrypoints, run conventions, operational rules, and non-ablation experiment notes for this repository.

For ablation-only history, see [docs/parametric_edge_ablation_log.md](./parametric_edge_ablation_log.md).

For cluster usage discipline and SLURM workflow details, see [docs/lab_cluster_practical_manual.md](./lab_cluster_practical_manual.md).

## Environment

The commands below assume the repo root is the working directory:

```bash
cd /home/viplab/jiaxin/parametric_edge_prediction
conda activate diffusers
```

Core training entrypoint:

```bash
python train.py --config <config.yaml>
python train.py --config <base_config.yaml> --override-config <override.yaml>
```

Config loading behavior:

- If `--config` points to a non-`default.yaml` file and no override is provided, `configs/parametric_edge/default.yaml` is merged first.
- If a config sets `_inherit_default: false`, it is treated as a standalone config and is loaded directly without merging `default.yaml`.
- If `--override-config` is used, the override is merged on top of the config loaded by `--config`.
- Historical overfit configs now live under `configs/parametric_edge/archived_overfit/` so the main config directory only contains current training entrypoints.

## Output Conventions

Main training runs write to experiment-specific directories under:

- `outputs/parametric_edge_training/`

Synthetic matching analysis writes to:

- `outputs/parametric_edge_training/current_sweep_analysis/`
- `outputs/parametric_edge_training/current_sweep_analysis/matching_synthetic/`

Dataset-side cache artifacts are stored separately under:

- `edge_data/HED-BSDS/cache/graph_polyline_v1_segments_xy_v5_anchor_consistent/`

These cache files are not source-controlled and should not be expected in Git history.

## Current Default Training

Purpose:

- Use the committed default config as the LAION pretraining recipe.

Config:

- [configs/parametric_edge/default.yaml](../configs/parametric_edge/default.yaml)
- [configs/parametric_edge/bsds_formal.yaml](../configs/parametric_edge/bsds_formal.yaml)

Current intended command:

```bash
python train.py --config configs/parametric_edge/default.yaml
```

Current runtime design:

- Train on LAION synthetic data with precomputed Bezier graph caches
- Validate and test on held-out LAION selections defined by `selection_offset`
- Use 4 GPUs with DDP in the main cluster recipe
- Keep both `best` and `last` checkpoints
- Use `ddp_find_unused_parameters_true`
- Use FP32 in the current cluster recipe
- Record train and validation visualizations periodically
- Current mainline model is `DABCurveDETR`, not the older `ParametricDETR`

Current status:

- The LAION cluster pretraining recipe is now the default config.
- The main configs now use the DAB-style curve model with:
  - `ResNet50`
  - `6 encoder / 6 decoder`
  - `hidden_dim: 256`
  - `num_queries: 300`
- The LAION mainline now uses effective batch size `64`.
- Auxiliary objectives are disabled by default:
  - `aux`
  - `DN`
  - `one-to-many`
  - `top-k`
  - `distinct`
- Following the 2026-03-24 overfit diagnosis, the main training configs disable:
  - `giou`
  - `sample`
  - `curve_distance`
- Historical BSDS training remains available as a separate path, but is no longer the default entrypoint.
- Use `configs/parametric_edge/bsds_formal.yaml` when you explicitly want the BSDS recipe.

## 2026-03-24 Summary

Main conclusions from today's debugging:

- The training mainline was reset from the older mixed `ParametricDETR` stack to a cleaner `DABCurveDETR` class.
- Single-image and 8-image overfit runs were used as the primary debugging tool.
- For overfit diagnostics:
  - `two_stage` query construction was confirmed to be a bad fit
  - learned DAB curve queries overfit cleanly
- `direction_invariant` itself was not the main problem; the earlier matcher implementation was.
- `giou` was identified as the main geometry term that prevented clean overfit, especially on long boundary edges.
- After removing `giou`, the previously problematic long edge in `183066_ann2` could be fit to the image boundary with near-zero endpoint error.
- The current default configs were then aligned to the stable subset that overfit supports:
  - `CE`
  - `ctrl`
  - `endpoint`

Main config changes made today:

- switched the main configs to `dab_curve_detr`
- moved to `ResNet50 + 6 encoder + 6 decoder`
- set `num_queries: 300`
- set LAION effective batch size to `64`
- disabled `dynamic_class_balance`
- removed `giou`, `sample`, and `curve_distance` from the active default training stack

## LAION Cluster Pretraining

Purpose:

- Run LAION-only pretraining on cluster from committed repo configs.

Primary config:

- [configs/parametric_edge/laion_pretrain_cluster.yaml](../configs/parametric_edge/laion_pretrain_cluster.yaml)

Fallback config:

- [configs/parametric_edge/laion_pretrain_cluster_2gpu.yaml](../configs/parametric_edge/laion_pretrain_cluster_2gpu.yaml)

Current cluster rules:

- Change training and data semantics in committed config files first.
- Use direct `sbatch`, not a repo-local wrapper launcher.
- If a temporary batch script is needed, keep it only on the cluster checkout.
- Keep cluster-only outputs, task folders, and logs on cluster.
- Use full FP32 for the current LAION pretraining path.
- Use the dedicated W&B project `laion_parametric_edge_prediction`.
- Prefer semantic run names and avoid hardware labels in future names.
- For `gbunchQ`, use `3-00:00:00` by default; for other partitions, use that partition's actual maximum time.

Example direct cluster submission:

```bash
sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH -J laion-pretrain-q256-eb256-lr5e5-fp32
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
python train.py --config configs/parametric_edge/laion_pretrain_cluster.yaml
EOF
```

## LAION Data Discovery Notes

Current behavior:

- LAION sample discovery writes `laion_entry_cache.txt` under the LAION data root.
- Cache reuse now probes only a small subset of cached entries at startup instead of existence-checking every record.
- In a single Python process, the parsed entry list is memoized by cache-file `mtime`, so train, val, and test discovery do not each re-read and re-parse the same large cache file.
- If the startup probe finds too many missing entries, the cache is treated as stale and the dataset list is rebuilt from the filesystem.
- During actual loading, if an individual sample disappears or is corrupted, the dataset falls back to another probed-valid sample with a bounded retry count instead of looping forever.

Current implication:

- Startup is no longer dominated by per-entry existence checks across the full cache.
- Startup is also no longer dominated by re-reading the same large text cache multiple times during one datamodule setup.
- Steady-state robustness is improved for partially damaged LAION datasets.
- Reading and parsing a very large text cache can still cost noticeable startup time even after removing the full validation sweep.

## Local Bottleneck Profile

Profile date:

- 2026-03-19

Purpose:

- Measure the current LAION cluster model architecture locally with a smaller per-GPU batch and identify whether the main bottleneck is dataloading, forward, matching/loss, or backward.

Effective test setup:

- Base config: [configs/parametric_edge/laion_pretrain_cluster_2gpu.yaml](../configs/parametric_edge/laion_pretrain_cluster_2gpu.yaml)
- Local dataset root retargeted to `/home/devdata/laion/edge_detection`
- Single RTX 3090, `batch_size: 2`, `num_workers: 8`, FP32, logging and checkpointing disabled
- Measured path: datamodule setup, first-batch fetch, steady-state fetch, host-to-device transfer, forward, loss, backward plus optimizer

Main measurements:

- Before the cache-probe change, local `datamodule.setup('fit')` was about `42.9s`
- After removing the full per-entry validation sweep, local `datamodule.setup('fit')` was about `39.8s`
- After adding in-process cache memoization, local repeated cache reads were about `7.08s`, then `0.13s`, then `0.13s`
- After both changes, local `datamodule.setup('fit')` was about `7.37s`
- First batch fetch was about `1.06s`
- Steady-state fetch was about `0.0002s/step`
- Host-to-device transfer was about `0.0005s/step`
- Forward was about `0.200s/step`
- Loss was about `0.363s/step`
- Backward plus optimizer was about `0.250s/step`
- Hungarian path was about `0.072s/step`
- Extra grouped cost-matrix construction outside the main matcher was about `0.116s/step`
- Peak memory was about `3.8GB`

Current conclusion:

- The steady-state bottleneck is not dataloader throughput.
- The main bottleneck is on the optimization side, especially `loss + backward`.
- Matching-related work is a major part of the loss cost.
- The dominant LAION startup overhead is now the first parse of the large text cache itself, not repeated parsing and not a full path-validation sweep.
- Moving discrete matching-only logic under `torch.no_grad()` is correct and was validated by a real local backward step after the change.
