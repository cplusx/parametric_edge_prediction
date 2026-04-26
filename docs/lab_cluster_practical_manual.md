# Lab / Cluster Practical Manual

This is the current operations guide for this repo.

## Connectivity Rules

- local machine can ssh to lab machines
- any lab machine can ssh to cluster
- cluster cannot ssh back to lab machines
- anything that moves from cluster to a lab machine must be relayed through a lab machine

Current common path:

```bash
log30
logcluster
```

## Current Repo Roots

Local:

- `/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction`

Lab machines:

- `lab30`: `/home/viplab/jiaxin/parametric_edge_prediction`
- `lab34`: `/data/jiaxin/parametric_edge_prediction`

Cluster:

- `/home/user/yc47434/parametric_edge_prediction`

## Current Data Roots

Lab30:

- `/home/devdata/laion/edge_detection`

Lab34:

- `/data/jiaxin/laion/edge_detection`

Cluster:

- `/home/user/yc47434/laion/edge_detection`

## Conda Env Mapping

Use `sam2` only for source-edge generation.

Use `diffusers` for:

- dataloader profiling
- dataset rendering
- DAB training
- cluster training submits
- lab34 training launches

Current rule of thumb:

- `sam2` = preprocessing from masks to source-edge
- `diffusers` = bezier cache consumption and all training-side code

## Current Preprocessing Flow

```text
images
  -> SAM2 masks
  -> source_edge PNGs
  -> bezier NPZs
  -> entry cache TXT
```

Current scripts:

- source-edge generation:
  - `scripts/generate_source_edges_from_sam2_batch.py`
- bezierization:
  - `scripts/generate_bezier_from_source_edges_batch.py`
- source-edge progress:
  - `scripts/check_source_edge_progress.py`
- bezier progress:
  - `scripts/check_bezierization_progress.py`
- sync completed results back to lab30:
  - `scripts/sync_completed_results_to_lab30.py`

## Current Training Flow

```text
bezier NPZs
  -> rewrite machine-local entry cache
  -> train.py
```

Required helper before training on a machine:

- `scripts/rewrite_bezier_entry_cache.py`

## Git Sync Workflow

Current helper:

- `scripts/sync_git_lab30_cluster.py`

Expected order:

1. commit locally
2. push locally
3. sync remotes
4. submit / launch

The sync script assumes local is the source of truth.

## Cluster Submission Rules

For this repo:

- use committed submit helpers under `scripts/submit_jobs/`
- cluster submission goes through `lab30 -> cluster`
- training jobs are `sbatch` jobs
- do not put `srun` inside the actual training sbatch body for launching the Python job
- use absolute paths in batch scripts
- use explicit conda activation in every batch script

Current cluster submit helpers:

- `scripts/submit_jobs/submit_curve_dab_cluster_train.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_line.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_emd_edgeprob05.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_emd_line.py`
- `scripts/submit_jobs/submit_endpoint_attach_cluster_train.py`
- `scripts/submit_jobs/submit_curve_dab_overfit8_cluster.py`

## Lab Launch Rule

Lab34 training is not a cluster job.

Current helper:

- `scripts/submit_jobs/submit_curve_dab_lab34_edgeprob05.py`

It writes a launcher on `lab34` and starts it with `nohup`.

## Resume Rule

Current training entrypoint supports:

```bash
python train.py --config <config> --resume
```

Meaning:

- if `default_root_dir/checkpoints/last.ckpt` exists, resume from it
- otherwise continue as a fresh run

## Monitoring / Debugging

Cluster queue:

```bash
squeue -u yc47434 -o '%.18i %.10P %.20j %.8T %.10M %.6D %R'
```

Detailed job record:

```bash
scontrol show job <jobid>
```

Attach helper function for an allocated job:

```bash
attach() {
  srun --jobid "$1" --overlap --pty bash
}
```

Use `attach <jobid>` from a cluster login shell.

## Current Stable Rules

- training is FP32 only
- v3 bezier caches are the maintained training input
- source-edge is preprocessing output, not training input
- submit scripts should print the actual launch content before submission
- if a workflow changes, update the matching doc in the same commit
