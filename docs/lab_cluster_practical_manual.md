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
- keep launch logic in committed repo configs and submit scripts
- if training settings change, edit the repo config locally, commit it, and let the cluster pull it
- keep cluster-only files limited to submitted batch scripts, logs, copied base configs, and runtime output directories

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
source /home/user/<cluster_user>/anaconda3/etc/profile.d/conda.sh
conda activate base
cd /home/user/<cluster_user>/<repo>
python your_script.py
```

Recommendations:

- write stdout/stderr to known locations
- use absolute paths
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
- Use `tmux` first.
- Probe before large jobs.
- Test GPU partitions for CPU-only jobs.
- Use absolute paths in SLURM scripts.
- Explicitly source conda inside jobs.
- Use a small number of large batch jobs for huge offline preprocessing.
- Monitor with both SLURM tools and output counts.
- Clean temporary artifacts after success.
