from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from textwrap import dedent

from remote_hosts import run_lab30, wrap_cluster_cmd


CLUSTER_REPO = "/home/user/yc47434/parametric_edge_prediction"
CLUSTER_RUN_ROOT = "/home/user/yc47434/cluster_runs/parametric_edge_prediction"
CONFIG_PATH = "configs/parametric_edge/laion_curve_pretrain_cluster_v3_2gpu.yaml"
SBATCH_PATH = f"{CLUSTER_REPO}/cluster_tasks/train_curve_dab_v3_2gpu.sbatch"


def build_sbatch() -> str:
    return dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH -J curve-dab-v3-2gpu
        #SBATCH -p gbunchQ
        #SBATCH --gres=gpu:2
        #SBATCH -c 16
        #SBATCH --mem=64G
        #SBATCH -t 3-00:00:00
        #SBATCH -o {CLUSTER_RUN_ROOT}/%x-%j.out
        #SBATCH -e {CLUSTER_RUN_ROOT}/%x-%j.err

        set -euo pipefail
        set +u
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
        conda activate diffusers
        set -u

        cd {CLUSTER_REPO}
        export PYTHONPATH={CLUSTER_REPO}${{PYTHONPATH:+:$PYTHONPATH}}
        python train.py --config {CONFIG_PATH}
        """
    )


def submit(sbatch_text: str) -> str:
    escaped = shlex.quote(sbatch_text)
    inner = (
        f"set -euo pipefail; "
        f"mkdir -p {CLUSTER_REPO}/cluster_tasks {CLUSTER_RUN_ROOT}; "
        f"cat > {SBATCH_PATH} <<'EOF'\n{sbatch_text}EOF\n"
        f"sbatch {SBATCH_PATH}"
    )
    proc = run_lab30(wrap_cluster_cmd(inner), timeout=120)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "sbatch submit failed")
    return proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or submit the 2-GPU cluster curve DAB sbatch job.")
    parser.add_argument("--submit", action="store_true", help="actually submit after printing")
    args = parser.parse_args()

    sbatch_text = build_sbatch()
    print("===== SBATCH BEGIN =====")
    print(sbatch_text, end="" if sbatch_text.endswith("\n") else "\n")
    print("===== SBATCH END =====")
    if not args.submit:
        return
    result = submit(sbatch_text)
    print(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
