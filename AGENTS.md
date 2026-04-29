# AGENTS.md

This file records the repo-specific operating rules that matter when working on cluster- or lab-launched training for `parametric_edge_prediction`.

## Canonical Sources

When cluster or training-launch behavior changes, keep these in sync in the same commit:

- `docs/training_and_submission.md`
- `docs/lab_cluster_practical_manual.md`
- `scripts/submit_jobs/`
- root launchers such as `run_lab34.sh`

If code and docs disagree, treat the current committed submit helpers and launch scripts as the source of truth, then update the docs.

## Connectivity And Machine Topology

- Local repo root: `/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction`
- `lab30` repo root: `/home/viplab/jiaxin/parametric_edge_prediction`
- `lab34` repo root: `/data/jiaxin/parametric_edge_prediction`
- Cluster repo root: `/home/user/yc47434/parametric_edge_prediction`

Network rules:

- local can ssh to lab machines
- lab machines can ssh to cluster
- cluster cannot ssh back to lab machines
- anything that must move from cluster to a lab machine has to be relayed through a lab machine

Operational consequence:

- cluster submits must go through `lab30 -> cluster`
- do not assume cluster work can be driven by a reverse connection from cluster back to local or lab

## Git And Sync Workflow

Before cluster submission or remote lab launch, the required order is:

1. commit locally
2. push locally
3. sync remotes
4. submit / launch

Use `scripts/sync_git_lab30_cluster.py` for the sync step.

Important sync assumptions:

- local is the source of truth
- the script expects the local git tree to be clean
- the script expects local `HEAD` to already be pushed to upstream
- remote dirty trees may be reset by the sync helper, so do not keep uncommitted work on `lab30`, `lab34`, or cluster

## Environments

- Use `sam2` only for preprocessing that turns masks into source-edge outputs.
- Use `diffusers` for bezier-cache consumption, training, cluster submit flows, and lab34 launch flows.

Training launchers and sbatch scripts should explicitly do:

- `source .../conda.sh`
- `conda activate diffusers`

Do not rely on an already-activated shell environment.

## Training Data And Cache Rules

Current training input is:

- bezier NPZs
- entry-cache TXT rewritten for the target machine

Current training does not consume source-edge PNGs directly.

Before training on any machine, rewrite the bezier entry cache with machine-local absolute paths via `scripts/rewrite_bezier_entry_cache.py`.

Current machine data roots:

- `lab30`: `/home/devdata/laion/edge_detection`
- `lab34`: `/data/jiaxin/laion/edge_detection`
- cluster: `/home/user/yc47434/laion/edge_detection`

Cluster sbatch scripts currently rewrite:

- input/output cache: `/home/user/yc47434/laion/edge_detection/laion_entry_cache_v3_bezier.txt`
- image root: `/home/user/yc47434/laion/edge_detection/edge_detection`
- bezier root: `/home/user/yc47434/laion/edge_detection/laion_edge_v3_bezier`

After rewriting the cache, keep the preflight check that verifies the first cache entry points to existing image and bezier files before launching `train.py`.

## Cluster Submission Requirements

For this repo, cluster training jobs must follow these rules:

- use the committed helpers under `scripts/submit_jobs/`
- submit through `lab30 -> cluster`
- use `sbatch`, not an interactive training launch
- do not put `srun` inside the sbatch body for the Python training command
- use absolute paths throughout the batch script
- explicitly activate `diffusers`
- `cd` into the cluster repo root before launch
- export `PYTHONPATH` with the repo root
- rewrite the bezier cache inside the job before training
- print the full sbatch content before actual submission

Current cluster submit helpers:

- `scripts/submit_jobs/submit_curve_dab_cluster_train.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_line.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_emd_edgeprob05.py`
- `scripts/submit_jobs/submit_curve_dab_cluster_train_emd_line.py`
- `scripts/submit_jobs/submit_endpoint_attach_cluster_train.py`
- `scripts/submit_jobs/submit_curve_dab_overfit8_cluster.py`

Current common cluster locations:

- repo root: `/home/user/yc47434/parametric_edge_prediction`
- run/log root: `/home/user/yc47434/cluster_runs/parametric_edge_prediction`
- sbatch output files should go to `%x-%j.out` and `%x-%j.err` under that run root

Current default resource pattern in committed 2-GPU training helpers:

- partition: `gbunchQ`
- GPUs: `2`
- CPUs: `32`
- memory: `64G`
- time: `3-00:00:00`

Current overfit helper exception:

- `submit_curve_dab_overfit8_cluster.py` uses `1` GPU, `16` CPUs, `48G`, `1-00:00:00`

## Resume Rule

If a run is meant to continue from `default_root_dir/checkpoints/last.ckpt`, launch with `--resume`.

Current repo convention:

- some cluster jobs intentionally include `--resume`
- some fresh-start helpers intentionally do not

Do not add or remove `--resume` casually; preserve the intended behavior of the specific run helper.

## Lab34 Is Separate From Cluster

`lab34` training is not a cluster job.

Use:

- committed launcher helper `scripts/submit_jobs/submit_curve_dab_lab34_edgeprob05.py`, or
- root launchers such as `run_lab34.sh`, `run_lab34_endpt.sh`, and `run_lab34_conditioned_curve.sh`

Lab34 launch behavior:

- write a shell launcher on `lab34`
- start it with `nohup`
- activate `diffusers`
- rewrite the bezier cache before training

`run_lab34*.sh` may source `run_lab34.local.env` for machine-local overrides such as W&B credentials. Keep those overrides out of tracked sync logic.

## Monitoring / Debugging

Useful cluster commands:

- `squeue -u yc47434 -o '%.18i %.10P %.20j %.8T %.10M %.6D %R'`
- `scontrol show job <jobid>`

Attach helper from a cluster login shell:

```bash
attach() {
  srun --jobid "$1" --overlap --pty bash
}
```

## Stable Repo Rules To Preserve

- training is FP32 only
- maintained training input is v3 bezier cache data
- source-edge files are preprocessing outputs, not direct training inputs
- cluster submit helpers should show the launch content before submission
- if launch workflow or machine assumptions change, update the matching docs in the same change
