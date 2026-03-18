# Lab Cluster Practical Manual

This is a practical manual for using the lab machine and the cluster through SLURM. It is intentionally general-purpose. It is not tied to one dataset or one cache-generation task.

## 1. Login path

Typical login chain:

```bash
log30
```

Then from the lab machine:

```bash
logcluster
```

If these aliases do not exist, inspect shell startup files:

```bash
grep -n 'log30\\|logcluster' ~/.zshrc ~/.bashrc 2>/dev/null
```

## 2. Use tmux before doing anything long

For anything interactive, long-running, or easy to disconnect from, start `tmux` first.

Create:

```bash
tmux new -s work
```

List:

```bash
tmux ls
```

Attach:

```bash
tmux attach -t work
```

Detach:

```bash
Ctrl-b d
```

## 3. Understand the cluster before choosing resources

Start with:

```bash
sinfo -o '%20P %10a %10l %6D %10c %10m %N'
sinfo -N -o '%20P %20N %8t %6c %10m %G'
```

What to look at:

- partition name
- time limit
- number of nodes
- CPUs per node
- memory per node
- whether the node is `idle`, `mix`, `alloc`, or `down`
- whether the node belongs to a GPU partition

Important habit:

- do not assume the CPU-only partition is always best
- do not assume GPU partitions are unusable for CPU-only jobs

Some GPU nodes may have much higher CPU counts and still accept CPU-only jobs.

## 4. Probe before the real run

Before launching a large real job, run a small probe.

The probe should test:

- can the partition accept your resource request
- does Python work on the compute node
- does your conda environment exist there
- is the repo path visible
- is the output directory writable

Example test-only submission:

```bash
sbatch --test-only -p gbunchQ -c 96 --mem=180G -t 02:00:00 --wrap='python -c "print(1)"'
```

Example tiny real probe script:

```bash
#!/bin/bash
#SBATCH -J probe
#SBATCH -p gbunchQ
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:10:00

set -e
source /home/user/<cluster_user>/anaconda3/etc/profile.d/conda.sh
conda env list
conda activate base
which python
python --version
```

## 5. Login node and compute node are not the same

This is one of the most common failure modes.

A job may fail immediately because:

- `python` exists on the login node but not on the compute node
- the conda environment name is wrong
- the compute node does not inherit the shell setup you expect

Safe pattern:

```bash
source /home/user/<cluster_user>/anaconda3/etc/profile.d/conda.sh
conda activate base
```

Do not rely on a shell alias or PATH being present inside the compute node.

## 6. Batch-first workflow for this repo

For `parametric_edge_prediction`, use **`sbatch` only**.

Do not use `srun` for this repository's training workflow.

Project-specific policy:

- debug with short `sbatch` probes instead of interactive `srun`
- keep launch logic in committed repo configs; do not add wrapper submit scripts for this repo
- if training settings change, edit the repo config locally, commit it, and let the cluster pull it
- keep temporary cluster task files under `cluster_tasks/` inside the repo and ignore that directory in Git
- keep cluster-only files limited to temporary batch scripts, logs, and runtime output directories

## 7. Writing reliable batch jobs

Recommended structure:

```bash
#!/bin/bash
#SBATCH -J myjob
#SBATCH -p <partition>
#SBATCH -c <cpus>
#SBATCH --mem=<memory>
#SBATCH -t <time>
#SBATCH -o /path/to/job.%j.out
#SBATCH -e /path/to/job.%j.err

set -e
set +u
source /home/user/<cluster_user>/anaconda3/etc/profile.d/conda.sh
conda activate base
set -u
cd /home/user/<cluster_user>/<repo>
python your_script.py
```

Recommendations:

- write stdout/stderr to known locations
- use absolute paths
- if the environment has activate hooks that assume unset shell vars are allowed, wrap `conda activate` with `set +u` before it and restore `set -u` after it
- make long preprocessing scripts skip outputs that already exist
- split large workloads into a small number of large jobs rather than many tiny jobs

## 8. Monitoring jobs

Current jobs:

```bash
squeue -u <cluster_user> -o '%.18i %.10P %.20j %.8T %.10M %.6D %R'
```

Finished job summary:

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,NodeList -P
```

Detailed job record:

```bash
scontrol show job <jobid>
```

When a job fails quickly, always inspect:

- `.out`
- `.err`
- `sacct`

before resubmitting.

## 9. Estimating remaining time

For large preprocessing jobs:

1. Count total inputs.
2. Count completed outputs.
3. Measure progress again after some time.
4. Convert to items/hour.
5. Estimate:

```text
remaining_time = remaining_items / current_items_per_hour
```

Be careful with two misleading phases:

- warm-up / environment setup
- skip-heavy phases where many outputs already exist

If your script logs both `ok` and `skip`, use `ok` for a more accurate throughput estimate.

## 10. Moving files between cluster and lab machine

Often the cleanest pattern is:

- compute on the cluster
- sync back from the lab machine

Example:

```bash
rsync -av <cluster_user>@<cluster_host>:/cluster/output/path/ /home/devdata/target/path/
```

For very long jobs, a watcher on the lab machine can:

- poll cluster progress every few minutes
- start `rsync` automatically when the final expected count is reached

## 11. Cleanup after the run

Good things to clean:

- probe scripts
- temporary `sbatch` scripts
- transient watcher scripts
- temporary progress logs
- debugging stdout/stderr files

Do **not** delete:

- final outputs
- synchronized destination data
- documentation or notes someone still needs

## 12. Practical rules of thumb

- Start with discovery, not assumptions.

## 13. Repo-Specific Notes For Parametric Edge

- Do not put hardware names like `h100` or `a40` into run names. Use model/data/batch semantics only.
- For this repo, temporary cluster task files belong under `cluster_tasks/` in the repo root, not under a separate external folder.
- LAION dataset discovery now supports a text entry cache under the LAION data root. If the cache file exists, reuse it; if it does not, create it once at startup.

## 14. Lessons From 2026-03-18

Mistakes made today:

- Requested `256G` for a 4-GPU job before checking the actual node layouts. That was unjustified and could unnecessarily restrict placement.
- Assumed the target machine type from the run name instead of checking the real node configuration.
- Reintroduced a repo-local wrapper submit script even though this repo should use direct `sbatch`.
- Used `set -u` around `conda activate` in a non-interactive batch shell, which broke activation through `ADDR2LINE` in the environment hook.

Correct workflow:

- Check `sinfo -N` or `scontrol show node` before choosing `--gres`, `-c`, and `--mem`.
- Distinguish scheduling failures from startup failures with `squeue`, `sacct`, `.out`, and `.err` before changing resources.
- Submit training with plain `sbatch`; if a temporary script is needed, place it in `cluster_tasks/` and keep it out of Git.
- In batch jobs, use `set +u`, then `source .../conda.sh`, then `conda activate ...`, then restore `set -u`.
- Use `srun --jobid ... --overlap` only as an attached monitoring step for a running allocation, not as the main launch path.
- Use `tmux` first.
- Probe before large jobs.
- Test GPU partitions for CPU-only jobs.
- Use absolute paths in SLURM scripts.
- Explicitly source conda inside jobs.
- Use a small number of large batch jobs for huge offline preprocessing.
- Monitor with both SLURM tools and output counts.
- Clean temporary artifacts after success.
