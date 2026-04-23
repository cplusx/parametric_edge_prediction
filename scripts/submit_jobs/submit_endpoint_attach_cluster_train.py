from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.remote_hosts import run_lab30, wrap_cluster_cmd  # noqa: E402


CLUSTER_REPO = "/home/user/yc47434/parametric_edge_prediction"
CLUSTER_RUN_ROOT = "/home/user/yc47434/cluster_runs/parametric_edge_prediction"
CONFIG_PATH = "configs/parametric_edge/laion_endpoint_attach_cluster_v3_2gpu.yaml"
SBATCH_PATH = f"{CLUSTER_REPO}/cluster_tasks/train_endpoint_attach_dab_v3_2gpu.sbatch"


def build_sbatch() -> str:
    return dedent(
        f"""\
        #!/usr/bin/env bash
        #SBATCH -J endpt-attach-v3-2gpu
        #SBATCH -p gbunchQ
        #SBATCH --gres=gpu:2
        #SBATCH -c 32
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
        python scripts/rewrite_bezier_entry_cache.py \\
          --input-cache /home/user/yc47434/laion/edge_detection/laion_entry_cache_v3_bezier.txt \\
          --output-cache /home/user/yc47434/laion/edge_detection/laion_entry_cache_v3_bezier.txt \\
          --image-root /home/user/yc47434/laion/edge_detection/edge_detection \\
          --bezier-root /home/user/yc47434/laion/edge_detection/laion_edge_v3_bezier
        python - <<'PY'
        from pathlib import Path

        cache = Path('/home/user/yc47434/laion/edge_detection/laion_entry_cache_v3_bezier.txt')
        first = next(line for line in cache.open('r', encoding='utf-8') if line.strip())
        batch, image_id, image_path, bezier_path = first.rstrip('\\n').split('\\t')
        if not Path(image_path).is_file():
            raise FileNotFoundError(f'cache image missing: {{image_path}}')
        if not Path(bezier_path).is_file():
            raise FileNotFoundError(f'cache bezier missing: {{bezier_path}}')
        print(f'cache preflight ok: {{batch}}/{{image_id}}')
        PY
        python train.py --config {CONFIG_PATH} --resume
        """
    )


def write_sbatch(sbatch_text: str) -> str:
    inner = (
        f"set -euo pipefail; "
        f"mkdir -p {CLUSTER_REPO}/cluster_tasks {CLUSTER_RUN_ROOT}; "
        f"cat > {SBATCH_PATH} <<'EOF'\n{sbatch_text}EOF\n"
        f"chmod +x {SBATCH_PATH}; "
        f"echo {SBATCH_PATH}"
    )
    proc = run_lab30(wrap_cluster_cmd(inner), timeout=120)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "sbatch write failed")
    return proc.stdout.strip()


def submit(sbatch_text: str) -> str:
    write_sbatch(sbatch_text)
    proc = run_lab30(wrap_cluster_cmd(f"sbatch {SBATCH_PATH}"), timeout=120)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "sbatch submit failed")
    return proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print, write, or submit the 2-GPU cluster endpoint attach DAB training job.")
    parser.add_argument("--write", action="store_true", help="write the sbatch file to cluster without submitting")
    parser.add_argument("--submit", action="store_true", help="write and submit after printing")
    args = parser.parse_args()

    if args.write and args.submit:
        raise ValueError("--write and --submit are mutually exclusive")

    sbatch_text = build_sbatch()
    print("===== SBATCH BEGIN =====")
    print(sbatch_text, end="" if sbatch_text.endswith("\n") else "\n")
    print("===== SBATCH END =====")
    if args.write:
        print(write_sbatch(sbatch_text))
    if args.submit:
        print(submit(sbatch_text))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
