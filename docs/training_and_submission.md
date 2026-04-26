# Training And Submission

This document describes the current training flow and launch conventions.

## Config Loading

Training entrypoint:

```bash
python train.py --config <config.yaml>
python train.py --config <config.yaml> --override-config <override.yaml>
python train.py --config <config.yaml> --resume
```

Config merge rule:

- `configs/parametric_edge/default.yaml` is loaded first
- selected config is merged on top of it
- `--override-config` is merged last
- if a config sets `_inherit_default: false`, it skips the default merge

Implementation:

- `misc_utils/config_utils.py`
- `train.py`

## Current Runtime Defaults

From `configs/parametric_edge/default.yaml` plus `train.py`:

- training precision is FP32 only
- `trainer.precision` must be `32-true`
- default `trainer.max_epochs` is `1000`
- `--resume` checks `trainer.default_root_dir/checkpoints/last.ckpt`
- runtime scaling can still expand `devices` and `accumulate_grad_batches` from config

## Current Dataset Assumption

Training no longer consumes source-edge PNGs directly.

Current training input path:

```text
bezier NPZs + entry_cache TXT -> dataloader -> training
```

Before launching training, the submit scripts rewrite the bezier entry cache so the paths match the target machine.

Helper:

- `scripts/rewrite_bezier_entry_cache.py`

## Current Active Training Configs

### Curve DAB

Cluster, random query init:

- `configs/parametric_edge/laion_curve_pretrain_cluster_v3_2gpu.yaml`

Cluster, line query init:

- `configs/parametric_edge/laion_curve_pretrain_cluster_v3_2gpu_line.yaml`

Cluster, EMD variants:

- `configs/parametric_edge/laion_curve_pretrain_cluster_v3_2gpu_emd_edgeprob05.yaml`
- `configs/parametric_edge/laion_curve_pretrain_cluster_v3_2gpu_emd_line.yaml`

Lab machines:

- `configs/parametric_edge/laion_curve_pretrain_lab30_v3_2gpu.yaml`
- `configs/parametric_edge/laion_curve_pretrain_lab34_v3_edgeprob05.yaml`

### Conditioned Curve DAB

Lab34:

- `configs/parametric_edge/laion_conditioned_curve_pretrain_lab34_v3_2gpu.yaml`

This branch uses:

- `model.arch: dab_cond_curve_detr`
- curve targets, like direct curve DAB
- merged endpoint condition points from the conditioned dataloader
- default init from a direct curve DAB checkpoint when
  - `model.conditioned_curve_init_enabled: true`
  - `model.conditioned_curve_init_checkpoint` is set

The endpoint-conditioning residual path is zero-initialized, so loading the
base curve DAB checkpoint starts from the same behavior as the direct curve
model before conditioning learns to contribute.

### Endpoint DAB

Point-only branch:

- `configs/parametric_edge/laion_endpoint_pretrain_lab34_v3_2gpu.yaml`

Attach branch:

- `configs/parametric_edge/laion_endpoint_attach_lab34_v3_2gpu.yaml`
- `configs/parametric_edge/laion_endpoint_attach_cluster_v3_2gpu.yaml`

## Query Initialization

Curve DAB now requires `model.curve_query_init_type` explicitly.

Supported values:

- `random`
  - every control point initialized independently
- `point`
  - each query is a repeated single point on a grid
- `line`
  - random endpoints with linear interpolation for middle control points

Implementation:

- `models/curve_query_initializers.py`

Conditioned curve init loader:

- `misc_utils/checkpoint_loading.py`

## Current Curve Matching / Loss Options

Curve branch supports:

- Chamfer distance
- EMD / Sinkhorn distance
- edge-probability cost term in matching

Main code:

- `models/matcher.py`
- `models/curve_distances.py`
- `models/losses/matched.py`

## Current Endpoint Matching / Loss Options

Point-only branch:

- Hungarian matching on point distance + class probability
- matched point loss uses point supervision

Attach branch:

- target mode: `data.endpoint_target_mode: attach`
- endpoint matcher can include incident-curve distance in matching
- matched endpoint loss can add attach distance to the incident GT curve set
- attach weighting depends on endpoint degree and loop-only status

Main code:

- `models/endpoint_matcher.py`
- `models/losses/endpoint_matched.py`
- `misc_utils/endpoint_target_utils.py`
- `edge_datasets/endpoint_attach_dataset.py`

## Submit Scripts

Cluster submit helpers:

- `scripts/submit_jobs/submit_curve_dab_cluster_train.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_line.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_emd_edgeprob05.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_emd_line.py`
- `scripts/submit_jobs/submit_endpoint_attach_cluster_train.py`
- `scripts/submit_jobs/submit_curve_dab_overfit8_cluster.py`

Lab launcher helper:

- `scripts/submit_jobs/submit_curve_dab_lab34_edgeprob05.py`

Current submit-script rules:

- cluster submit scripts print the full `sbatch` content before submission
- cluster scripts go through `lab30 -> cluster`
- lab34 launcher writes a shell script on `lab34` and starts it with `nohup`
- training jobs should use `--resume` when the run directory is intended to continue from `last.ckpt`

Current direct lab34 launch scripts in repo root:

- `run_lab34.sh`
- `run_lab34_endpt.sh`
- `run_lab34_conditioned_curve.sh`

## Sync Rules Before Submission

Git sync helper:

- `scripts/sync_git_lab30_cluster.py`

Remote helper:

- `scripts/remote_hosts.py`

Operational rule:

- local commit first
- push first
- then sync remote repos
- then submit from the helper script

## Current Output Roots

Typical training output roots:

- cluster:
  - `/home/user/yc47434/cluster_runs/parametric_edge_prediction`
- lab34:
  - `/data/jiaxin/parametric_edge_prediction/outputs`
- lab30:
  - config-specific output root under the local repo or machine data root

## What Was Removed From The Docs

These older doc assumptions are no longer correct:

- source-edge as the direct training input
- stale machine batch status as if it were current documentation
- old architecture figures that no longer match the maintained code path
