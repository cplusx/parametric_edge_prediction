from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.remote_hosts import run_lab34  # noqa: E402


LAB34_REPO = "/data/jiaxin/parametric_edge_prediction"
LAB34_DATA_ROOT = "/data/jiaxin/laion/edge_detection"
CONFIG_PATH = "configs/parametric_edge/laion_curve_pretrain_lab34_v3_edgeprob05.yaml"
LAUNCHER_PATH = f"{LAB34_REPO}/outputs/launch_curve_dab_lab34_edgeprob05.sh"
LAUNCH_LOG = f"{LAB34_REPO}/outputs/launch_curve_dab_lab34_edgeprob05.log"


def build_launcher(*, resume: bool = False) -> str:
    resume_arg = " --resume" if resume else ""
    return dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        set +u
        source /home/viplab/anaconda3/etc/profile.d/conda.sh
        conda activate diffusers
        set -u

        cd {LAB34_REPO}
        export PYTHONPATH={LAB34_REPO}${{PYTHONPATH:+:$PYTHONPATH}}
        python scripts/rewrite_bezier_entry_cache.py \\
          --input-cache {LAB34_DATA_ROOT}/laion_entry_cache_v3_bezier.txt \\
          --output-cache {LAB34_DATA_ROOT}/laion_entry_cache_v3_bezier.txt \\
          --image-root {LAB34_DATA_ROOT} \\
          --bezier-root {LAB34_DATA_ROOT}/laion_edge_v3_bezier
        python train.py --config {CONFIG_PATH}{resume_arg}
        """
    )


def submit(launcher_text: str) -> str:
    remote_cmd = dedent(
        f"""\
        set -euo pipefail
        mkdir -p {LAB34_REPO}/outputs
        cat > {LAUNCHER_PATH} <<'EOF'
        {launcher_text}EOF
        chmod +x {LAUNCHER_PATH}
        nohup bash {LAUNCHER_PATH} > {LAUNCH_LOG} 2>&1 < /dev/null &
        echo $!
        """
    )
    proc = run_lab34(remote_cmd, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "lab34 submit failed")
    return proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or launch the lab34 edge-prob-matching curve DAB run.")
    parser.add_argument("--submit", action="store_true", help="actually launch on lab34 after printing")
    parser.add_argument("-r", "--resume", action="store_true", help="resume from default_root_dir/checkpoints/last.ckpt if available")
    args = parser.parse_args()

    launcher_text = build_launcher(resume=args.resume)
    print("===== LAUNCHER BEGIN =====")
    print(launcher_text, end="" if launcher_text.endswith("\n") else "\n")
    print("===== LAUNCHER END =====")
    if not args.submit:
        return
    pid = submit(launcher_text)
    print(f"PID {pid}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
